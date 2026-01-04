import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('medical_examination.csv')

# Add overweight column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# Normalize cholesterol and glucose
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


def draw_cat_plot():
    # Create a copy for the categorical plot
    df_cat = df.copy()
    
    # Melt the dataframe
    df_cat = pd.melt(df_cat, 
                     id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],
                     var_name='variable',
                     value_name='value')
    
    # Create the categorical plot
    fig = sns.catplot(data=df_cat,
                      kind='bar',
                      x='variable',
                      y='value',
                      hue='cardio',
                      height=5,
                      aspect=2)
    
    # Set labels
    fig.set_axis_labels('variable', 'total')
    
    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    # Clean the data for heatmap
    df_heat = df.copy()
    
    # Remove rows where diastolic pressure is greater than systolic
    df_heat = df_heat[df_heat['ap_lo'] <= df_heat['ap_hi']]
    
    # Remove height outside 2.5th to 97.5th percentile
    height_min = df_heat['height'].quantile(0.025)
    height_max = df_heat['height'].quantile(0.975)
    df_heat = df_heat[(df_heat['height'] >= height_min) & (df_heat['height'] <= height_max)]
    
    # Remove weight outside 2.5th to 97.5th percentile
    weight_min = df_heat['weight'].quantile(0.025)
    weight_max = df_heat['weight'].quantile(0.975)
    df_heat = df_heat[(df_heat['weight'] >= weight_min) & (df_heat['weight'] <= weight_max)]
    
    # Calculate correlation matrix
    corr = df_heat.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create figure and plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, 
                mask=mask,
                annot=True,
                fmt='.1g',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                ax=ax)
    
    fig.savefig('heatmap.png')
    return fig
