# ===== PDT - SPECIFIC USER DIALOGUES =====
def Warning(s: str):
    print('PDT WARNING: {}'.format(s))


def Info(s: str, indent_level: int = 0):
    ind_txt = '    ' * indent_level
    offset_spaces = '          '
    print('PDT INFO: {}{}'.format(ind_txt, s.replace('\n', '\n{}{}'.format(offset_spaces, ind_txt))))


def error_plot(**kwargs):

    import matplotlib.pyplot as plt

    if all(plot_param in kwargs for plot_param in ['x', 'y', 'xlbl', 'ylbl']):
        x = kwargs.pop('x')
        y = kwargs.pop('y')
        xlbl = kwargs.pop('xlbl')
        ylbl = kwargs.pop('ylbl')
        ax0_is_plot = True
    else:
        ax0_is_plot = False

    if 'info_d' in kwargs:
        info_d = kwargs.pop('info_d')
    else:
        info_d = None

    if 'info_d2' in kwargs:
        info_d2 = kwargs.pop('info_d2')
    else:
        info_d2 = None

    fig = plt.figure(figsize=(12, 8))
    axes = fig.subplots(nrows=1, ncols=2)

    def get_txt_start_params():
        txt_xstart = 0.05
        txt_ystart = 0.95
        txt_xinc = 0.0
        txt_yinc = -0.04
        return txt_xstart, txt_ystart, txt_xinc, txt_yinc

    def plot_info_dict(ax, info_dict: dict):
        x_txt, y_txt, x_inc, y_inc = get_txt_start_params()
        for header, details in info_dict.items():
            txt = '{}: {}'.format(header, details)
            lines = [txt[i:i + 48] for i in range(0, len(txt), 48)]
            ax.text(x_txt, y_txt, "\n   ".join(lines), ha='left', va='top')
            y_txt += y_inc * len(lines)
            x_txt += x_inc * len(lines)

    if ax0_is_plot:
        axes[0].grid(True)
        axes[0].set_title('PDT Troubleshooting Plot')
        axes[0].set_xlabel(xlbl)
        axes[0].set_ylabel(ylbl)
        axes[0].plot(x, y)

        axes[1].set_title('PDT Troubleshooting Info')
        axes[1].axes.xaxis.set_visible(False)
        axes[1].axes.yaxis.set_visible(False)
        if info_d is not None:
            plot_info_dict(ax=axes[1], info_dict=info_d)

    else:
        for ax in axes:
            ax.set_title('PDT Troubleshooting Info')
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        if info_d is not None:
            plot_info_dict(ax=axes[0], info_dict=info_d)
            if info_d2 is not None:
                plot_info_dict(ax=axes[1], info_dict=info_d2)

    return fig


class Error(Exception):
    def __init__(self, s: str, errplot: bool = False, **errplot_kwargs):
        if errplot:
            error_plot(**errplot_kwargs)
        super().__init__('\nPDT ERROR: {}'.format(s.replace('\n', '\n           ')))
