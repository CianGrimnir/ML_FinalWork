#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime
import folium
import webbrowser


df = pd.read_csv("dublinbikes_20200101_20200401.csv")

eval_df = df.copy()
eval_df['TIME'] = pd.to_datetime(eval_df['TIME'])
eval_df['date'] = eval_df['TIME'].dt.floor('D')
eval_df = eval_df.groupby(['STATION ID', 'date', 'STATUS', 'ADDRESS']).agg({'AVAILABLE BIKES': 'sum', 'AVAILABLE BIKE STANDS': 'sum'})
eval_df = eval_df.reset_index(level=[0, 1, 2, 3])

top_address = eval_df.groupby('ADDRESS')['AVAILABLE BIKE STANDS'].sum().nlargest(20).index.tolist()
top_df = pd.DataFrame(columns=eval_df.columns)
for address in top_address:
    res = eval_df[eval_df['ADDRESS'] == address]
    top_df = top_df.append(res)
# normal plot show
top_df.set_index('date', inplace=True)
# top_df.groupby('ADDRESS')['AVAILABLE BIKES'].plot(Legend=True)
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(2,2)
fig, ax = plt.subplots(2, 2)


# minimize the gap between time interval
def plot_graph(ax, top_address, label):
    for address in top_address:
        ax.plot(top_df[top_df['ADDRESS'] == address].index, top_df[top_df['ADDRESS'] == address]['AVAILABLE BIKE STANDS'], label=address)
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax.grid(True)
    ax.set_title(f'top {label} dublin station against cycle count')
    ax.set_xlabel("Day")
    ax.set_ylabel("Total Cycle Usage")
    ax.legend(loc="upper left")
    ax.set_xlim([datetime.date(2020, 2, 1), datetime.date(2020, 3, 1)])
    return ax


ax[0, 0] = plot_graph(ax[0, 0], top_address[0:5], '5')
ax[0, 1] = plot_graph(ax[0, 1], top_address[5:10], '5-10')
ax[1, 0] = plot_graph(ax[1, 0], top_address[10:15], '10-15')
ax[1, 1] = plot_graph(ax[1, 1], top_address[15:20], '15-20')
fig.tight_layout()

fig.subplots_adjust(hspace=0.2, wspace=0.2)

plt.show()


loc_df = df[['ADDRESS', 'LATITUDE', 'LONGITUDE']]
loc_df = loc_df.drop_duplicates()
city_latitude = float((loc_df.loc[loc_df['ADDRESS'].isin(['Townsend Street'])])['LATITUDE'])
city_longitude = float((loc_df.loc[loc_df['ADDRESS'].isin(['Townsend Street'])])['LONGITUDE'])
title = 'Dublin Bikes Stand Q1 - 2020'
title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(title)
m = folium.Map([city_latitude, city_longitude], zoom_start=14)
for add in loc_df['ADDRESS']:
    stand_lat = float((loc_df.loc[loc_df['ADDRESS'].isin([add])])['LATITUDE'])
    stand_lon = float((loc_df.loc[loc_df['ADDRESS'].isin([add])])['LONGITUDE'])
    folium.Marker(location=[stand_lat, stand_lon], popup=f"{add}",
                  icon=folium.Icon(color='darkred', icon='bicycle', prefix='fa')).add_to(m)
m.get_root().html.add_child(folium.Element(title_html))
m.save("stand_location.html")
webbrowser.open("stand_location.html")

# Add additional features to dataset
df1 = df.copy()
df1['STATUS'] = df1['STATUS'].astype('category')
df1['LAST UPDATED'] = pd.to_datetime(df1['LAST UPDATED'])
df1['TIME'] = pd.to_datetime(df1['TIME'])
df1['formatted_time'] = df1['TIME'].dt.floor('h')
df1['day_of_week'] = df1['formatted_time'].dt.strftime('%A')
df1['day_of_month'] = df1['formatted_time'].dt.strftime('%d').astype(np.int64)
df1['hour'] = df1['formatted_time'].dt.strftime('%H').astype(np.int64)
df1['month'] = df1['formatted_time'].dt.strftime('%m').astype(np.int64)
df1['week'] = df1['formatted_time'].dt.strftime('%w').astype(np.int64)
numeric_columns = df1.select_dtypes(['int64']).columns
print(df1[numeric_columns].describe().T)
print(df1.select_dtypes(['category']).describe().T)
print(df1.dtypes)

# Descriptive statistics for all the DateTime features
print(df1.select_dtypes(['datetime64[ns]']).describe(datetime_is_numeric=True).T)
# Descriptive statistics for all the categorial features
print(df1['AVAILABLE BIKES'].describe().T)
print(df1['AVAILABLE BIKE STANDS'].describe().T)
# df1['TIME'] = df1['TIME'].dt.floor('Min')
df1['STATUS'] = df1['STATUS'].astype('category')
print(df1['STATUS'].describe().T)

# box plot for available bikes
color = {'boxes': 'DarkGreen', 'whiskers': 'DarkOrange', 'medians': 'DarkBlue', 'caps': 'Gray'}

df1['AVAILABLE BIKES'].plot(color=color, kind='box', figsize=(20, 20), title='Available bike usage', legend=True, grid=True)
plt.show()

# Histograms for all continuous features within the dataset
hist_df = df1.copy()
hist_df = hist_df[['TIME', 'AVAILABLE BIKES', 'AVAILABLE BIKE STANDS', 'hour', 'day_of_month']]
numeric_columns = hist_df.select_dtypes(['int64']).columns
# numeric_columns = numeric_columns.drop('STATION ID')
# Histograms for all continuous features within the dataset
hist_df[numeric_columns].hist(figsize=(20, 20))
hist_df['day_of_month'].value_counts().sort_index().plot()
# extract features from 2nd Feb to 1st Apr for station - Merrion Square/Grangegorman
ts = pd.to_datetime('2020/01/01')
te = pd.to_datetime('2020/04/01')
mask = (df1['TIME'] >= ts) & (df1['TIME'] <= te)
pd.options.mode.chained_assignment = None
# df['TIME'].max() - df['TIME'].min()
suburb_point = "Merrion Square South"
suburb_df = df1.loc[mask]
suburb_df = suburb_df[suburb_df['ADDRESS'] == suburb_point]
suburb_dataset = suburb_df[['TIME', 'AVAILABLE BIKES']]
numeric_columns = suburb_df.select_dtypes(['int64']).columns
print(suburb_df[numeric_columns].describe().T)

suburb_dataset['date'] = suburb_dataset['TIME'].dt.floor('T')
suburb_dataset = suburb_dataset.reset_index()
suburb_dataset.drop('TIME', axis=1, inplace=True)
time = suburb_dataset['date']
bikes = suburb_dataset['AVAILABLE BIKES']
# plot 2 week data for Merrion Square South region
fig = plt.figure()
ax = fig.add_subplot(111)
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.scatter(time, bikes, color="red", marker=".", label='bike count')
ax.set_xlabel("date")
ax.set_ylabel("bike available")
fig.autofmt_xdate()
ax.grid(True)
ax.set_title(f'Bike usage for {suburb_point}')
ax.set_xlim([datetime.date(2020, 1, 1), datetime.date(2020, 4, 2)])
plt.show()

