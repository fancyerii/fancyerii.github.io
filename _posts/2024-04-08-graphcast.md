---
layout:     post
title:      "Graphcast: How to Get Things Done" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - Graphcast
---

本文是论文[Graphcast: How to Get Things Done](https://towardsdatascience.com/graphcast-how-to-get-things-done-f2fd5630c5fb)的翻译。

<!--more-->

**目录**
* TOC
{:toc}


天气预报是一个非常复杂的问题。数值天气预报（NWP）模型，如天气研究与预报（WRF）模型，已被用于解决这个问题，然而，有时准确性和精确度被发现不足。

由于其复杂性，从数据科学家到气象工程师，都对解决方案产生了兴趣和追求。虽然已经找到了一些解决方案，但一致性和一致性仍然缺乏。解决方案因地区而异，从山区到高原，从沼泽到苔原。根据我个人的经验，我相信也是其他人的经验，天气预报被发现是一个棘手的问题。引用某位虾类亿万富翁的话：


>这就像一盒巧克力，你永远不知道会得到什么。

最近，Deepmind发布了一个新工具：Graphcast，这是一个用于更快更准确的全球天气预报的人工智能模型，试图让这个特定的巧克力包更美味、更高效。在谷歌的 TPU v4 机器上，使用 Graphcast，可以在不到一分钟内以 0.25 度空间分辨率获取预测结果。它解决了使用传统方法进行预测时可能面临的许多问题：

* 预测一次性生成所有坐标的预测值，
* 根据坐标编辑逻辑现在已经是多余的，
* 令人惊叹的效率和响应时间。

不那么令人费解的是，使用上述工具获取预测所需的数据准备工作。
 

然而，请不用担心，我将成为你在黑暗和阴郁的铠甲中的骑士，并在本文中解释准备和格式化数据以及最终使用Graphcast获取预测所需的步骤。

注：现在“人工智能”这个词的使用方式很像漫威电影中“量子”的使用方式。

获取预测是一个可以分为以下几个部分的过程：

* 获取输入数据。
* 创建目标。
* 创建外部数据。
* 处理和格式化数据为合适的格式。
* 将它们汇总并进行预测。

Graphcast 表示，使用当前天气数据和 6 小时前的数据，可以预测未来 6 小时的情况。举例来说明：

* 如果需要预测的时间是：2024 年 01 月 01 日 18:00，
* 那么要提供的输入数据是：2024 年 01 月 01 日 12:00 和 2024 年 01 月 01 日 06:00。

重要的是要注意，2024 年 01 月 01 日 18:00 将是第一个预测的时间。Graphcast 还可以额外获取 10 天的数据，每个预测之间间隔 6 小时。因此，可以获取预测的其他时间戳为：

* 2024 年 01 月 02 日 00:00、06:00、12:00、18:00，
* 2024 年 01 月 03 日 00:00、06:00，以此类推，
* 直至 2024 年 01 月 10 日 06:00、12:00。

总结一下，可以使用两个时间戳的输入预测 40 个时间戳的数据。

## 假设和重要参数

对于我在本文中将要呈现的代码，我已经给定了一些参数的以下值，这些参数决定了你能够多快地获得预测以及使用的内存量。

* 输入时间戳：2024 年 01 月 01 日 6:00、12:00。
* 第一个预测时间戳：2024 年 01 月 01 日 18:00。
* 预测数量：4。
* 空间分辨率：1 度。
* 压力水平：13。

以下是导入所需包、初始化用于输入和预测的字段所需的数组以及其他有用的变量的代码。

```python
import cdsapi
import datetime
import functools
from graphcast import autoregressive, casting, checkpoint, data_utils as du, graphcast, normalization, rollout
import haiku as hk
import isodate
import jax
import math
import numpy as np
import pandas as pd
from pysolar.radiation import get_radiation_direct
from pysolar.solar import get_altitude
import pytz
import scipy
from typing import Dict
import xarray

client = cdsapi.Client() # Making a connection to CDS, to fetch data.

# The fields to be fetched from the single-level source.
singlelevelfields = [
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                        '2m_temperature',
                        'geopotential',
                        'land_sea_mask',
                        'mean_sea_level_pressure',
                        'toa_incident_solar_radiation',
                        'total_precipitation'
                    ]

# The fields to be fetched from the pressure-level source.
pressurelevelfields = [
                        'u_component_of_wind',
                        'v_component_of_wind',
                        'geopotential',
                        'specific_humidity',
                        'temperature',
                        'vertical_velocity'
                    ]

# The 13 pressure levels.
pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Initializing other required constants.
pi = math.pi
gap = 6 # There is a gap of 6 hours between each graphcast prediction.
predictions_steps = 4 # Predicting for 4 timestamps.
watts_to_joules = 3600
first_prediction = datetime.datetime(2024, 1, 1, 18, 0) # Timestamp of the first prediction.
lat_range = range(-180, 181, 1) # Latitude range.
lon_range = range(0, 360, 1) # Longitude range.

# A utility function used for ease of coding.
# Converting the variable to a datetime object.
def toDatetime(dt) -> datetime.datetime:
    if isinstance(dt, datetime.date) and isinstance(dt, datetime.datetime):
        return dt
    
    elif isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
        return datetime.datetime.combine(dt, datetime.datetime.min.time())
    
    elif isinstance(dt, str):
        if 'T' in dt:
            return isodate.parse_datetime(dt)
        else:
            return datetime.datetime.combine(isodate.parse_date(dt), datetime.datetime.min.time())
```

## 输入

在机器学习中，为了获得一些预测，你必须向机器学习模型提供一些数据，以便它生成预测结果。例如，当预测一个人是否是蝙蝠侠时，输入数据可能是：

* 他们睡眠的时间有多长？
* 他们的脸上有没有晒斑？
* 他们在清晨的会议中是否睡觉？
* 他们的净资产有多少？

同样地，Graphcast 也需要一些特定的输入，我们通过其 Python 库：[cdsapi](https://pypi.org/project/cdsapi/) 从 [CDS](https://cds.climate.copernicus.eu/cdsapp#!/home) 获取这些输入。目前，数据发布者使用的是知识共享署名 4.0 许可证，这意味着只要给予原作者以适当的信用，任何人都可以复制、分发、传输和改编该作品。

然而，在使用 cdsapi 获取数据之前，需要进行身份验证，其步骤由 CDS 提供并且相当简单。

假设你现在已经通过了 CDS 的审核，输入可以通过以下步骤来创建：

* 获取单层值：这些值依赖于坐标和时间。其中一个需要的输入字段是 total_precipitation_6hr。顾名思义，它是从特定时间戳开始前 6 小时的降雨量累积。因此，我们不仅要获取两个输入时间戳的值，还需要获取从 2024 年 01 月 01 日 00:00 到 12:00 的时间戳的值。
* 获取压力层值：除了依赖于坐标，它们还取决于压力层。因此，在请求数据时，我们会提到我们需要数据的压力层。在这种情况下，我们只获取两个输入时间戳的值。
* 合并单层和压力值：在上述数据基础上进行基于时间、纬度和经度的内部合并操作。
* 整合年份和日期进度：除了单层和压力字段外，还需要向输入数据中添加四个字段：year_progress_sin、year_progress_cos、day_progress_sin 和 day_progress_cos。可以使用 graphcast 包提供的函数来完成这一步骤。

其他小步骤包括：

* 在从 CDS 获取数据后重命名列，因为 CDS 输出了天气变量的缩写形式。
* 将单层数据的geopotential重命名为geopotential_at_surface，因为压力层具有相同的字段名称。
* 在从 graphcast 获取进度值后，使用数学函数计算 sin 和 cos 值。
* 将纬度重命名为 lat、经度重命名为 lon，并引入另一个索引：batch，其值为 0。

创建输入数据的代码如下所示。

```python

# Getting the single and pressure level values.
def getSingleAndPressureValues():
    
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': singlelevelfields,
            'grid': '1.0/1.0',
            'year': [2024],
            'month': [1],
            'day': [1],
            'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00'],
            'format': 'netcdf'
        },
        'single-level.nc'
    )
    singlelevel = xarray.open_dataset('single-level.nc', engine = scipy.__name__).to_dataframe()
    singlelevel = singlelevel.rename(columns = {col:singlelevelfields[ind] for ind, col in enumerate(singlelevel.columns.values.tolist())})
    singlelevel = singlelevel.rename(columns = {'geopotential': 'geopotential_at_surface'})

    # Calculating the sum of the last 6 hours of rainfall.
    singlelevel = singlelevel.sort_index()
    singlelevel['total_precipitation_6hr'] = singlelevel.groupby(level=[0, 1])['total_precipitation'].rolling(window = 6, min_periods = 1).sum().reset_index(level=[0, 1], drop=True)
    singlelevel.pop('total_precipitation')
    
    client.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': pressurelevelfields,
            'grid': '1.0/1.0',
            'year': [2024],
            'month': [1],
            'day': [1],
            'time': ['06:00', '12:00'],
            'pressure_level': pressure_levels,
            'format': 'netcdf'
        },
        'pressure-level.nc'
    )
    pressurelevel = xarray.open_dataset('pressure-level.nc', engine = scipy.__name__).to_dataframe()
    pressurelevel = pressurelevel.rename(columns = {col:pressurelevelfields[ind] for ind, col in enumerate(pressurelevel.columns.values.tolist())})

    return singlelevel, pressurelevel

# Adding sin and cos of the year progress.
def addYearProgress(secs, data):

    progress = du.get_year_progress(secs)
    data['year_progress_sin'] = math.sin(2 * pi * progress)
    data['year_progress_cos'] = math.cos(2 * pi * progress)

    return data

# Adding sin and cos of the day progress.
def addDayProgress(secs, lon:str, data:pd.DataFrame):

    lons = data.index.get_level_values(lon).unique()
    progress:np.ndarray = du.get_day_progress(secs, np.array(lons))
    prxlon = {lon:prog for lon, prog in list(zip(list(lons), progress.tolist()))}
    data['day_progress_sin'] = data.index.get_level_values(lon).map(lambda x: math.sin(2 * pi * prxlon[x]))
    data['day_progress_cos'] = data.index.get_level_values(lon).map(lambda x: math.cos(2 * pi * prxlon[x]))
    
    return data

# Adding day and year progress.
def integrateProgress(data:pd.DataFrame):
        
    for dt in data.index.get_level_values('time').unique():
        seconds_since_epoch = toDatetime(dt).timestamp()
        data = addYearProgress(seconds_since_epoch, data)
        data = addDayProgress(seconds_since_epoch, 'longitude' if 'longitude' in data.index.names else 'lon', data)

    return data

# Adding batch field and renaming some others.
def formatData(data:pd.DataFrame) -> pd.DataFrame:
        
    data = data.rename_axis(index = {'latitude': 'lat', 'longitude': 'lon'})
    if 'batch' not in data.index.names:
        data['batch'] = 0
        data = data.set_index('batch', append = True)
    
    return data

if __name__ == '__main__':

    values:Dict[str, xarray.Dataset] = {}
    
    single, pressure = getSingleAndPressureValues()
    values['inputs'] = pd.merge(pressure, single, left_index = True, right_index = True, how = 'inner')
    values['inputs'] = integrateProgress(values['inputs'])
    values['inputs'] = formatData(values['inputs'])
```

## Targets

有11个预测字段：

* u_component_of_wind,
* v_component_of_wind,
* geopotential,
* specific_humidity,
* temperature,
* vertical_velocity,
* 10m_u_component_of_wind,
* 10m_v_component_of_wind,
* 2m_temperature,
* mean_sea_level_pressure,
* total_precipitation. 
 
目标是在每个坐标、预测时间戳和压力水平上传递一个基本为空的xarray，用于所有的预测字段。以下是执行此操作的代码。

```python
# Includes the packages imported and constants assigned.
# The functions created for the inputs also go here.

predictionFields = [
                        'u_component_of_wind',
                        'v_component_of_wind',
                        'geopotential',
                        'specific_humidity',
                        'temperature',
                        'vertical_velocity',
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                        '2m_temperature',
                        'mean_sea_level_pressure',
                        'total_precipitation_6hr'
                    ]

# Creating an array full of nan values.
def nans(*args) -> list:
    return np.full((args), np.nan)

# Adding or subtracting time.
def deltaTime(dt, **delta) -> datetime.datetime:
    return dt + datetime.timedelta(**delta)

def getTargets(dt, data:pd.DataFrame):
    
    # Creating an array consisting of unique values of each index.
    lat, lon, levels, batch = sorted(data.index.get_level_values('lat').unique().tolist()), sorted(data.index.get_level_values('lon').unique().tolist()), sorted(data.index.get_level_values('level').unique().tolist()), data.index.get_level_values('batch').unique().tolist()
    time = [deltaTime(dt, hours = days * gap) for days in range(4)]

    # Creating an empty dataset using latitude, longitude, the pressure levels and each prediction timestamp.
    target = xarray.Dataset({field: (['lat', 'lon', 'level', 'time'], nans(len(lat), len(lon), len(levels), len(time))) for field in predictionFields}, coords = {'lat': lat, 'lon': lon, 'level': levels, 'time': time, 'batch': batch})

    return target.to_dataframe()

if __name__ == '__main__':

    # The code for creating inputs will be here.

    values['targets'] = getTargets(first_prediction, values['inputs'])
```


## Forcings
与目标一样，forcings也包含每个坐标和预测时间戳的值，但不包括压力水平。forcings中的字段包括：

* total_incident_solar_radiation,
* year_progress_sin,
* year_progress_cos,
* day_progress_sin,
* day_progress_cos.

需要注意的是，上述值是相对于预测时间戳分配的。与处理输入时一样，年和日进度仅取决于时间戳，并且太阳辐射是从单层源获取的。然而，由于正在进行预测，即获取未来的值，所以在forcings的情况下，太阳值不会在CDS数据集中可用。因此，我们使用pysolar库模拟太阳辐射值。

```python
# Includes the packages imported and constants assigned.
# The functions created for the inputs and targets also go here.

# Adding a timezone to datetime.datetime variables.
def addTimezone(dt, tz = pytz.UTC) -> datetime.datetime:
    dt = toDatetime(dt)
    if dt.tzinfo == None:
        return pytz.UTC.localize(dt).astimezone(tz)
    else:
        return dt.astimezone(tz)

# Getting the solar radiation value wrt longitude, latitude and timestamp.
def getSolarRadiation(longitude, latitude, dt):
        
    altitude_degrees = get_altitude(latitude, longitude, addTimezone(dt))
    solar_radiation = get_radiation_direct(dt, altitude_degrees) if altitude_degrees > 0 else 0

    return solar_radiation * watts_to_joules

# Calculating the solar radiation values for timestamps to be predicted.
def integrateSolarRadiation(data:pd.DataFrame):
    
    dates = list(data.index.get_level_values('time').unique())
    coords = [[lat, lon] for lat in lat_range for lon in lon_range]
    values = []
    
    # For each data, getting the solar radiation value at a particular coordinate.
    for dt in dates:
        values.extend(list(map(lambda coord:{'time': dt, 'lon': coord[1], 'lat': coord[0], 'toa_incident_solar_radiation': getSolarRadiation(coord[1], coord[0], dt)}, coords)))
  
    # Setting indices.
    values = pd.DataFrame(values).set_index(keys = ['lat', 'lon', 'time'])
      
    # The forcings dataset will now contain the solar radiation values.
    return pd.merge(data, values, left_index = True, right_index = True, how = 'inner')

def getForcings(data:pd.DataFrame):
  
    # Since forcings data does not contain batch as an index, it is dropped.
    # So are all the columns, since forcings data only has 5, which will be created.
    forcingdf = data.reset_index(level = 'level', drop = True).drop(labels = predictionFields, axis = 1)
    
    # Keeping only the unique indices.
    forcingdf = pd.DataFrame(index = forcingdf.index.drop_duplicates(keep = 'first'))

    # Adding the sin and cos of day and year progress.
    # Functions are included in the creation of inputs data section.
    forcingdf = integrateProgress(forcingdf)

    # Integrating the solar radiation values.
    forcingdf = integrateSolarRadiation(forcingdf)

    return forcingdf

if __name__ == '__main__':

    # The code for creating inputs and targets will be here.

    values['forcings'] = getForcings(values['targets'])
```

## 后处理输入、目标和forcings

现在Graphcast的三大支柱已经建立，我们进入最后阶段。就像NBA总决赛中赢得了3场比赛一样，我们现在继续进行最紧张和最具挑战性的部分，将其完成。

就像科比·布莱恩特曾经说过的那样，

>工作还没有结束。

在xarray中，有两种主要类型的数据：

* 坐标，即索引：纬度、经度、时间等等；以及
* 数据变量，即列：陆海掩模、地势高度等等。
* 数据变量包含的每个值都有特定的坐标与之对应。

这些坐标是数据变量值所依赖的坐标。举个例子，看看我们自己的数据，

* 陆海掩模仅依赖于纬度和经度，这些是它的坐标。
* 地势高度的坐标包括批次、纬度、经度、时间和压力水平。
* 在鲜明对比的情况下，但却具有合理性，geopotential_at_surface的坐标是纬度和经度。

因此，在我们继续预测天气之前，我们确保每个数据变量都被分配到了正确的坐标上，其代码如下所示。


```python
# Includes the packages imported and constants assigned.
# The functions created for the inputs, targets and forcings also go here.

# A dictionary created, containing each coordinate a data variable requires.
class AssignCoordinates:
    
    coordinates = {
                    '2m_temperature': ['batch', 'lon', 'lat', 'time'],
                    'mean_sea_level_pressure': ['batch', 'lon', 'lat', 'time'],
                    '10m_v_component_of_wind': ['batch', 'lon', 'lat', 'time'],
                    '10m_u_component_of_wind': ['batch', 'lon', 'lat', 'time'],
                    'total_precipitation_6hr': ['batch', 'lon', 'lat', 'time'],
                    'temperature': ['batch', 'lon', 'lat', 'level', 'time'],
                    'geopotential': ['batch', 'lon', 'lat', 'level', 'time'],
                    'u_component_of_wind': ['batch', 'lon', 'lat', 'level', 'time'],
                    'v_component_of_wind': ['batch', 'lon', 'lat', 'level', 'time'],
                    'vertical_velocity': ['batch', 'lon', 'lat', 'level', 'time'],
                    'specific_humidity': ['batch', 'lon', 'lat', 'level', 'time'],
                    'toa_incident_solar_radiation': ['batch', 'lon', 'lat', 'time'],
                    'year_progress_cos': ['batch', 'time'],
                    'year_progress_sin': ['batch', 'time'],
                    'day_progress_cos': ['batch', 'lon', 'time'],
                    'day_progress_sin': ['batch', 'lon', 'time'],
                    'geopotential_at_surface': ['lon', 'lat'],
                    'land_sea_mask': ['lon', 'lat'],
                }

def modifyCoordinates(data:xarray.Dataset):
    
    # Parsing through each data variable and removing unneeded indices.
    for var in list(data.data_vars):
        varArray:xarray.DataArray = data[var]
        nonIndices = list(set(list(varArray.coords)).difference(set(AssignCoordinates.coordinates[var])))
        data[var] = varArray.isel(**{coord: 0 for coord in nonIndices})
    data = data.drop_vars('batch')

    return data

def makeXarray(data:pd.DataFrame) -> xarray.Dataset:
    
    # Converting to xarray.
    data = data.to_xarray()
    data = modifyCoordinates(data)

    return data

if __name__ == '__main__':

    # The code for creating inputs, targets and forcings will be here.

    values = {value:makeXarray(values[value]) for value in values}
```


## 使用Graphcast进行预测

经过计算、处理和组装输入、目标和forcings，现在是进行预测的时候了。

我们现在需要模型权重和标准化统计文件，[这些文件](https://console.cloud.google.com/storage/browser/dm_graphcast;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)由Deepmind提供。

需要下载的文件包括：

* stats/diffs_stddev_by_level.nc，
* stats/stddev_by_level.nc，
* stats/mean_by_level.nc和
* params/GraphCast_small — ERA5 1979–2015 — resolution 1.0 — pressure levels 13 — mesh 2to5 — precipitation input and output.npz。

上述文件相对于预测文件的相对路径如下所示。保持这种结构很重要，这样可以成功导入和读取所需的文件。

```
.
├── prediction.py
├── model
    ├── params
        ├── GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz
    ├── stats
        ├── diffs_stddev_by_level.nc
        ├── mean_by_level.nc
        ├── stddev_by_level.nc
```


通过Deepmind提供的预测代码，以上所有功能最终以以下代码片段进行预测。


```python
# Includes the packages imported and constants assigned.
# The functions created for the inputs, targets and forcings also go here.

with open(r'model/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz', 'rb') as model:
    ckpt = checkpoint.load(model, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

with open(r'model/stats/diffs_stddev_by_level.nc', 'rb') as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()

with open(r'model/stats/mean_by_level.nc', 'rb') as f:
    mean_by_level = xarray.load_dataset(f).compute()

with open(r'model/stats/stddev_by_level.nc', 'rb') as f:
    stddev_by_level = xarray.load_dataset(f).compute()
    
def construct_wrapped_graphcast(model_config:graphcast.ModelConfig, task_config:graphcast.TaskConfig):
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(predictor, diffs_stddev_by_level = diffs_stddev_by_level, mean_by_level = mean_by_level, stddev_by_level = stddev_by_level)
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing = True)
    return predictor

@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template = targets_template, forcings = forcings)

def with_configs(fn):
    return functools.partial(fn, model_config = model_config, task_config = task_config)

def with_params(fn):
    return functools.partial(fn, params = params, state = state)

def drop_state(fn):
    return lambda **kw: fn(**kw)[0]

run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

class Predictor:

    @classmethod
    def predict(cls, inputs, targets, forcings) -> xarray.Dataset:
        predictions = rollout.chunked_prediction(run_forward_jitted, rng = jax.random.PRNGKey(0), inputs = inputs, targets_template = targets, forcings = forcings)
        return predictions

if __name__ == '__main__':

    # The code for creating inputs, targets, forcings & processing will be here.

    predictions = Predictor.predict(values['inputs'], values['targets'], values['forcings'])
    predictions.to_dataframe().to_csv('predictions.csv', sep = ',')
```


## 结论

在上面，我提供了每个流程的代码：

* 创建输入、目标和forcings，
* 将上述数据处理为可行格式，最后
* 将它们组合在一起并进行预测。
* 在执行过程中，将所有流程整合在一起以实现无缝的实现很重要。

为了简化起见，我已经上传了[代码](https://github.com/abhinavyesss/graphcast-predict)以及docker镜像和容器文件，可以用于创建一个执行预测程序的环境。

在天气预测的宇宙中，我们目前有像Accuweather、IBM、多个气象模型这样的贡献者。Graphcast证明是一个有趣的、在许多情况下更有效的增加。然而，它也有一些远非最佳的属性。在一个难得的思考时刻，我得出了以下见解：

* 与其他天气预测服务相比，Graphcast效率更高，速度更快，在几分钟内为整个世界提供预测。
* 这使得通过API为数百个地理位置进行数百次调用变得多余。
* 然而，要在几分钟内完成上述操作，需要拥有一台非常强大的计算机，要么是Google TPU v4或更好。这并非易得。即使选择使用来自AWS、Google或Azure的虚拟机，成本也可能增加。
* 目前，没有规定可以使用小地理区域或坐标子集的数据并对其进行预测。始终需要所有坐标的数据。
* CDS提供的数据有5天的延迟期，这意味着在‘x’日期，CDS只能提供到‘x-5’日期的数据。这使得未来天气预测变得有些复杂，因为必须在进行未来预测之前覆盖延迟期。

需要注意的是，Graphcast是天气预测领域的一个相对较新的增加，将会进行改进和补充以提高访问便利性和可用性。鉴于其在效率和性能方面的领先优势，他们肯定会利用这一优势。

资源：

* [Graphcast演示代码](https://colab.research.google.com/drive/1X9WcRis_PC_DyuHYIiUwKWCAIr8T-4Pd#scrollTo=Sd99tPA3TBa4)
* [模型权重和统计文件](https://console.cloud.google.com/storage/browser/dm_graphcast)
* [论文](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/Learning_skillful_medium-range_global_weather_forecasting.pdf)
* [文章](https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/)
* [CDS](https://cds.climate.copernicus.eu/#!/home)

祝您在数据科学的旅程中一切顺利，谢谢您的阅读 :)
