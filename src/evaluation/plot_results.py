import locale
import matplotlib.pyplot as plt
import seaborn as sns

locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

def plot_metric_boxplot(runs, metric):
    display_names = {
        'training_strategy': 'Training Strategy',
        'loss_configuration': 'Loss Configuration',
    }

    _runs  = runs.copy()
    _runs['training_strategy'] = _runs['training_strategy'].astype(str)
    _runs = _runs[_runs['training_strategy']!= 'pretrain']
    _runs = _runs.rename(columns= display_names)

    ax = sns.catplot(
        y=metric,
        x='Training Strategy',
        hue='Loss Configuration',
        col='Loss Configuration',
        data=_runs,
        kind='box',
        height=4,
        aspect=0.4,
        legend=False,
    )

    duration_ticks = [1, 3, 10, 30, 100]
    duration_tick_labels = [str(tick) + ' min' for tick in duration_ticks]

    plot_params = {
        'regret': {
            'ylabel': 'Regret (€)',
            'yticklabels': [f'{tick:n} €' for tick in ax.axes[0][0].get_yticks()],
        },
        'train_duration_per_epoch': {
            'ylabel': 'Training Time (min/epoch)',
            'ylim':(1, 100),
            'yscale':'log',
            'yticks':duration_ticks,
            'yticklabels':duration_tick_labels
        },
    }

    ax.set(
        xlabel=' ',
        **plot_params[metric]
    )
    ax.set_titles(template='{col_name}')

    plt.show()



def plot_metrics_scatter_plot(runs, results):
    col_name_map = {'training_strategy': 'Training Strategy', 'loss_configuration': 'Loss Configuration'}
    _runs, _results  = runs.copy(), results.copy()
    _runs, _results = (df.rename(columns= col_name_map) for df in (_runs, _results))

    scatterplot_params = {
        'x': 'train_duration_per_epoch',
        'y': 'regret',
        'hue': 'Loss Configuration',
        'hue_order': ['MAE', 'MAE + SPO+', 'MAE + Cov-e', 'MAE + Corr-f'],
        'style': 'Training Strategy',
        'style_order': ['train', 'pretrain', 'finetune'],
        'markers': ['D', 'X', 'o'],
    }
    ax = sns.scatterplot(
        data=_runs,
        legend=False,
        s=75,
        alpha=0.4,
        **scatterplot_params,
    )
    ax = sns.scatterplot(
        data=_results,
        edgecolor='black',
        s=100,
        **scatterplot_params,
    )
    x_ticks = [1, 3, 10, 30, 100]
    x_tick_labels = [str(tick) + ' min' for tick in x_ticks]

    ax.set(
        xlabel='Training Time (min/epoch)',
        xlim=(1, 100),
        xscale='log',
        xticks=x_ticks,
        xticklabels=x_tick_labels,
        ylabel='Regret (€)',
        yticklabels=[f'{tick:n} €' for tick in ax.get_yticks()],
    )
    ax.figure.set_size_inches(5, 4)  # Width = 10, Height = 5
    sns.move_legend(ax, "upper right", bbox_to_anchor=(1, 1), fontsize='small', ncols=2, columnspacing=0.5)
    sns.set_theme(font='Times New Roman')

    return ax


def plot_correlation_with_regret(runs):
    _runs = runs.set_index(['dataset', 'training_strategy', 'loss_configuration', 'seed', 'regret'])[
        ['mae', 'rmse', 'cov_e', 'corr_f']].stack()
    _runs = _runs.reset_index()
    _runs = _runs.rename(columns={
        'level_5': 'metric',
        0: 'value',
        'training_strategy': 'Training Strategy',
        'loss_configuration': 'Loss Configuration'
    })

    ax = sns.relplot(
        x='value',
        y='regret',
        hue='Loss Configuration',
        # col='loss_configuration',
        row='dataset',
        col='metric',
        data=_runs,
        kind='scatter',
        facet_kws={'sharey': 'row', 'sharex': False},
        height=2.5,
        hue_order=['MAE', 'MAE + SPO+', 'MAE + Cov-e', 'MAE + Corr-f'],
        style='Training Strategy',
        style_order=['train', 'pretrain', 'finetune'],
        markers=['D', 'X', 'o'],
    )
    ax.set_titles(template='{row_name} | {col_name}')
    return ax