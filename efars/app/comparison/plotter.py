import datetime
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas
from matplotlib.ticker import FuncFormatter
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def strip_chars(someString):
    """ Provides """
    someString = someString.replace(" ", "-")
    someString = someString.replace("_", "-")
    return "".join(c for c in someString if c in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-")


class Plotter():
    def __init__(self, run_config):
        self.run_config = run_config

    def calculate_quality_metrics_results(self):
        """ Calculates the quality metrics for the evaluation """
        # calc necessar maximms
        f1_score_max = 0
        failure_max = 0
        duration_max = 0
        for run, fetches_df in self.fetch_metrics.items():
            f1_score_max = max(f1_score_max, fetches_df['f1 score'].max())
            failure_max = max(failure_max, fetches_df['failure_indice'].max())
            duration_max = max(duration_max, fetches_df['time_elapsed'].max())

        # increase maximums to avoid tigh layout
        f1_score_max = f1_score_max * 1.05
        failure_max = failure_max * 1.05
        duration_max = duration_max * 1.05

        for run, fetches_df in self.fetch_metrics.items():
            label_suffix = "{0}".format(run)
            self.scores_per_run(fetches_df, label_suffix,
                                f1_score_max, failure_max, duration_max)

    def calculate_quality_metrics_boxplots(self):
        """ Calculates the quality metrics as boxplots for the evaluation """
        bp_f1_df = pandas.DataFrame()
        bp_fail_df = pandas.DataFrame()
        for run, fetches_df in self.fetch_metrics.items():
            bp_f1_df[run] = fetches_df['f1 score']
            bp_fail_df[run] = fetches_df['failure_indice']

        # show box plot:
        bp = bp_f1_df.boxplot()
        bp.set_ylabel('F1 score')
        label = "box-plot-f1"
        save_path = os.path.join(
            self.run_config.evaluation_plots_folder_path, "{0}.pdf".format(strip_chars(label)))

        bp = bp_fail_df.boxplot()
        bp.set_ylabel('Average Amount of Failed Requests')
        label = "box-plot-failure"
        save_path = os.path.join(
            self.run_config.evaluation_plots_folder_path, "{0}.pdf".format(strip_chars(label)))
        plt.savefig(save_path)
        plt.close()

    def colorbrewer2_colors(self, i):
        colors = ["#e41a1c", "#377eb8", "#4daf4a",
                  "#984ea3", "#ff7f00", "#ffff33", "#a65628"]
        select_idx = i % len(colors)
        return colors[select_idx]

    def scores_per_run(self, df, label_suffix, f1_score_max, failure_max, duration_max):
        """ plots the f1 score and the failed requests """
        df = df.set_index("time_elapsed")
        x_axis = df.index

        # define axis
        f1_axis = df['f1 score']
        failure_axis = df['failure_indice']

        # plot f1 metrics and failed fetches in one diagram:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        self.plot_f1_with_failures(
            fig, ax1, x_axis, f1_axis, failure_axis, f1_score_max, failure_max, duration_max)
        # add grid and select tight layout
        ax1.grid(which='major', axis='y', linestyle='--')
        fig.tight_layout()

        ax1.patch.set_visible(False)  # hide the 'canvas'
        label = "f1-failure-{0}".format(label_suffix)
        save_path = os.path.join(
            self.run_config.evaluation_plots_folder_path, "{0}.pdf".format(strip_chars(label)))
        plt.savefig(save_path)

    def plot_f1_with_failures(self, fig, ax1, x_axis, f1_axis, failure_axis, f1_score_max, failure_max, duration_max):
        # plot f1 metrics and failed fetches in one diagram:
        ax1.plot(x_axis, f1_axis, '-', zorder=10,
                 color=self.colorbrewer2_colors(1), linewidth=2)
        ax1.set_ylabel('F1 Score')
        ax1.set_xlabel('Duration (h:mm:ss)')
        ax1.set_ylim(bottom=0, top=f1_score_max)
        ax1.set_xlim(left=duration_max*-0.01, right=duration_max)
        
        formatter = matplotlib.ticker.FuncFormatter(format_time_ticks)
        ax1.xaxis.set_major_formatter(formatter)

        # plot the failures as bars
        bar_width = (len(x_axis) * 1)
        ax2 = ax1.twinx()
        ax2.set_ylim(bottom=0, top=failure_max)
        #ax2.set_xlim(left=0, right=duration_max )
        ax2.bar(x_axis, failure_axis, bar_width,
                color=self.colorbrewer2_colors(0), edgecolor='none', zorder=1)
        ax2.set_ylabel('Average Amount of Failed Requests')
        ax1.set_zorder(ax2.get_zorder()+1)  # put ax in front of ax2

    def calculate_performance_results(self):
        # CPU usage is measured in "user jiffies (1/100 th of a second)" (aka 10ms)
        for container, data_per_run_dic in self.monitor_data.items():
            label_suffix = "{0}".format(container)
            if container.lower().startswith('cpu'):
                self.cpu_usage_per_run(data_per_run_dic, label_suffix)
                self.cpu_usage_percent_per_run(data_per_run_dic, label_suffix)
            if container.lower().startswith('memory'):
                self.mem_usage_per_run(data_per_run_dic, label_suffix)

    def calculate_performance_boxplots(self):
        # CPU usage is measured in "user jiffies (1/100 th of a second)" (aka 10ms)
        for container, data_per_run_dic in self.monitor_data.items():
            label_suffix = "{0}".format(container)
            if container.lower().startswith('cpu'):
                cpu_bp_df = pandas.DataFrame()
                avgcpu_bp_df = pandas.DataFrame()
                for run_type, df in data_per_run_dic.items():
                    cpu_bp_df[run_type] = (
                        df['cpu_stats_cpu_usage_total_usage'] / 100)
                    avgcpu_bp_df[run_type] = (df['avg_cpu'])
                bp = cpu_bp_df.boxplot()
                bp.set_ylabel('CPU Usage in Seconds')
                label = "box-plot-cpu-usage-{0}".format(label_suffix)
                save_path = os.path.join(
                    self.run_config.evaluation_plots_folder_path, "{0}.pdf".format(strip_chars(label)))
                plt.savefig(save_path)
                avg_bp = avgcpu_bp_df.boxplot()
                avg_bp.set_ylabel('CPU Usage in %')
                avg_bp.yaxis.set_major_formatter(
                    FuncFormatter('{0:0.1%}'.format))
                label = "box-plot-cpu-percentage-{0}".format(label_suffix)
                save_path = os.path.join(
                    self.run_config.evaluation_plots_folder_path, "{0}.pdf".format(strip_chars(label)))
                plt.savefig(save_path)
            if container.lower().startswith('memory'):
                mem_bp_df = pandas.DataFrame()
                for run_type, df in data_per_run_dic.items():
                    mem_bp_df[run_type] = df['usage'].apply(
                        lambda x: x / 1024 / 1024)
                bp = mem_bp_df.boxplot()
                bp.set_ylabel('Memory Usage in MiB')
                label = "box-plot-mem-usage-{0}".format(label_suffix)
                save_path = os.path.join(
                    self.run_config.evaluation_plots_folder_path, "{0}.pdf".format(strip_chars(label)))
                plt.savefig(save_path)
            plt.close()

    def cpu_usage_percent_per_run(self, data_per_run_dic, label_suffix):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylabel('CPU Usage in %')
        ax.set_xlabel('Duration (h:mm:ss)')

        x_max = 0
        y_max = 0
        i = -1
        for run_type, df in data_per_run_dic.items():
            i += 1
            df = df.set_index("time_elapsed")
            x_axis = df.index
            # https://github.com/moby/moby/issues/16849
            y = df['avg_cpu']
            ax.plot(x_axis, y, 'o-', color=self.colorbrewer2_colors(i),
                    label="{0}".format(run_type))

            # set the axis according to max values:
            y_max = max(y_max, y.max())
            x_max = max(x_max, x_axis.max())

        ax.set_ylim(bottom=0, top=y_max*1.05)
        ax.set_xlim(left=0, right=x_max*1.05)

        formatter = matplotlib.ticker.FuncFormatter(format_time_ticks)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(FuncFormatter('{0:0.1%}'.format))

        # Other formatting stuff
        plt.grid(which='major', axis='both', linestyle='--')
        plt.legend()
        label = "cpu-usage-percent-{0}".format(label_suffix)
        save_path = os.path.join(
            self.run_config.evaluation_plots_folder_path, "{0}.pdf".format(strip_chars(label)))
        plt.savefig(save_path)

    def cpu_usage_per_run(self, data_per_run_dic, label_suffix):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylabel('CPU Usage in Seconds')
        ax.set_xlabel('Duration (h:mm:ss)')

        x_max = 0
        y_max = 0
        i = -1
        for run_type, df in data_per_run_dic.items():
            i += 1
            df = df.set_index("time_elapsed")
            x_axis = df.index
            # convert to jiffies to seconds
            y = df['cpu_stats_cpu_usage_total_usage'] / 100
            ax.plot(x_axis, y, 'o-', color=self.colorbrewer2_colors(i),
                    label="{0}".format(run_type))
            # set the axis according to max values:
            y_max = max(y_max, y.max())
            x_max = max(x_max, x_axis.max())

        ax.set_ylim(bottom=0, top=y_max*1.05)
        ax.set_xlim(left=0, right=x_max*1.05)

        formatter = matplotlib.ticker.FuncFormatter(format_time_ticks)
        ax.xaxis.set_major_formatter(formatter)
        # Other formatting stuff
        plt.grid(which='major', axis='both', linestyle='--')
        plt.legend()
        label = "cpu-usage-{0}".format(label_suffix)
        save_path = os.path.join(
            self.run_config.evaluation_plots_folder_path, "{0}.pdf".format(strip_chars(label)))
        plt.savefig(save_path)

    def mem_usage_per_run(self, data_per_run_dic, label_suffix):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylabel('Memory Usage in MiB')
        ax.set_xlabel('Duration (h:mm:ss)')

        x_max = 0
        y_max = 0
        i = -1
        for run_type, df in data_per_run_dic.items():
            i += 1
            df = df.set_index("time_elapsed")
            x_axis = df.index
            y = df['usage'].apply(lambda x: x / 1024 / 1024)
            ax.plot(x_axis, y, 'o-', color=self.colorbrewer2_colors(i),
                    label="{0}".format(run_type))
            # set the axis according to max values:
            y_max = max(y_max, y.max())
            x_max = max(x_max, x_axis.max())

        # format the time according to https://stackoverflow.com/questions/15240003/matplotlib-intelligent-axis-labels-for-timedelta
        formatter = matplotlib.ticker.FuncFormatter(format_time_ticks)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_ylim(bottom=0, top=y_max*1.05)
        ax.set_xlim(left=0, right=x_max*1.05)

        # Other formatting stuff
        plt.grid(which='major', axis='both', linestyle='--')
        plt.legend()
        label = "mem-usage-{0}".format(label_suffix)
        save_path = os.path.join(
            self.run_config.evaluation_plots_folder_path, "{0}.pdf".format(strip_chars(label)))
        plt.savefig(save_path)

    def load_monitor_csvs(self):
        final_result = {}
        csv_monitor_folder = os.path.join(
            self.run_config.evaluation_csv_folder_path, "monitor")
        groups = [f for f in os.listdir(csv_monitor_folder) if not os.path.isfile(
            os.path.join(csv_monitor_folder, f))]
        for group in groups:
            group_fp = os.path.join(csv_monitor_folder, group)
            files = [f for f in os.listdir(
                group_fp) if os.path.isfile(os.path.join(group_fp, f))]
            if not group in final_result.keys():
                final_result[group] = {}
            for csv_file in files:
                run_type = csv_file[:-4]
                load_path = os.path.join(group_fp, csv_file)
                final_result[group][run_type] = pandas.read_csv(load_path)
        self.monitor_data = final_result

    def load_fetch_csvs(self):
        final_result = {}
        csv_fetch_folder = os.path.join(
            self.run_config.evaluation_csv_folder_path, "fetch")
        fetches = [f for f in os.listdir(csv_fetch_folder) if os.path.isfile(
            os.path.join(csv_fetch_folder, f))]
        for csv_file in fetches:
            cat_run = csv_file[:-4]
            load_path = os.path.join(csv_fetch_folder, csv_file)
            final_result[cat_run] = pandas.read_csv(load_path)
        self.fetch_metrics = final_result

# format the time according to https://stackoverflow.com/questions/15240003/matplotlib-intelligent-axis-labels-for-timedelta
def format_time_ticks(value, pos):
    "Formats this data to a human readable string"
    timedlt = datetime.timedelta(seconds=value)
    return str(timedlt)
