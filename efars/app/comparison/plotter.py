import datetime
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas
from matplotlib.ticker import FuncFormatter
from pandas.plotting import register_matplotlib_converters
import numpy

register_matplotlib_converters()


class Plotter():
    def __init__(self, run_config):
        self.run_config = run_config

    def calculate_quality_metrics_results(self):
        """ Calculates the quality metrics for the evaluation """
        # calc necessar maximms
        f1_score_max = 0
        failure_max = 0
        duration_max = 0
        for run, fetches_df in self.receiver_metrics.items():
            f1_score_max = max(f1_score_max, fetches_df['f1 score'].max())
            failure_max = max(failure_max, fetches_df['failure_indice'].max())
            duration_max = max(duration_max, fetches_df['time_elapsed'].max())

        for run, pdf in self.provider_metrics.items():
            failure_max = max(failure_max, pdf['provisions_failed_cnt'].max())

        # increase maximums to avoid tigh layout
        f1_score_max = f1_score_max * 1.05
        failure_max = failure_max * 1.05
        duration_max = duration_max * 1.05

        for run, rdf in self.receiver_metrics.items():
            pdf = self.provider_metrics[run]
            label_suffix = "{0}".format(run)
            self.plot_quality_with_failures(rdf, pdf,  label_suffix,
                                            f1_score_max, failure_max, duration_max)

    def plot_quality_with_failures(self, rdf, pdf, label_suffix, f1_score_max, failure_max, duration_max):
        """ plots the f1 score and the failed requests """

        rdf['provisions_failed_cnt'] = pdf['provisions_failed_cnt']
        rdf = rdf.set_index("time_elapsed")

        x_axis = rdf.index

        # define axis
        f1_axis = rdf['f1 score']
        receiver_errors_axis = rdf['failure_indice']
        provider_errors_axis = rdf['provisions_failed_cnt']

        # plot f1 metrics and failed fetches in one diagram:
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(111)

        self.plot_f1_with_failures(
            fig, ax1, x_axis, f1_axis, provider_errors_axis, receiver_errors_axis, f1_score_max, failure_max, duration_max)

        ax1.patch.set_visible(False)  # hide the 'canvas'

        fig.tight_layout()
        label = "f1-failure-{0}".format(label_suffix)
        save_path = os.path.join(
            self.run_config.evaluation_plots_folder_path, "{0}.pdf".format(strip_chars(label)))
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def plot_f1_with_failures(self, fig, ax1, x_axis, f1_axis, provider_f_axis, receiver_f_axis, f1_score_max, failure_max, duration_max):
        # plot f1 metrics and failed fetches in one diagram:
        ax1.plot(x_axis, f1_axis, '-', zorder=10,
                 color=self.colorbrewer2_colors(0), linewidth=2)
        ax1.set_ylabel('F1 Score')
        ax1.set_xlabel('Duration (h:mm:ss)')
        ax1.set_ylim(bottom=0, top=f1_score_max)

        formatter = matplotlib.ticker.FuncFormatter(format_time_ticks)
        ax1.xaxis.set_major_formatter(formatter)
        # add grid and select tight layout
        ax1.grid(which='major', axis='y', linestyle='--')

        # plot the failures as bars
        bar_width = (len(x_axis) * 5)
        ax2 = ax1.twinx()
        ax2.set_ylim(bottom=0, top=failure_max)

        ax2.bar(x_axis, receiver_f_axis, bar_width,
                color=self.colorbrewer2_colors(1), edgecolor='none', zorder=1, label='Failed Requests')
        ax2.bar(x_axis, provider_f_axis, width=bar_width*0.4,
                color=self.colorbrewer2_colors(2), edgecolor='none', zorder=2, label='Failed Provisions')
        ax2.set_ylabel('Average Failures')

        ax1.set_zorder(ax2.get_zorder()+1)  # put ax in front of ax2

    def calculate_quality_metrics_boxplots(self):
        """ Calculates the quality metrics as boxplots for the evaluation """
        bp_f1_df = pandas.DataFrame()
        bp_receiver_fail_df = pandas.DataFrame()
        bp_provider_fail_df = pandas.DataFrame()
        for run, fetches_df in self.receiver_metrics.items():
            pdf = self.provider_metrics[run]
            bp_f1_df[run] = fetches_df['f1 score']
            bp_receiver_fail_df[run] = fetches_df['failure_indice']
            bp_provider_fail_df[run] = pdf['provisions_failed_cnt']

        # show box plot:
        self.__boxplot(bp_f1_df, 'F1 score', "f1-bp")
        self.__boxplot(bp_receiver_fail_df,
                       'Average Failed Requests', "receiver-failure-bp")
        self.__boxplot(bp_provider_fail_df,
                       'Average Failed Provisions', "provider-failure-bp")

    def colorbrewer2_colors(self, i):
        colors = ["#e41a1c", "#377eb8", "#4daf4a",
                  "#984ea3", "#ff7f00", "#ffff33", "#a65628"]
        select_idx = i % len(colors)
        return colors[select_idx]

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
                # box plot CPU Usage in Seconds
                self.__boxplot(cpu_bp_df, 'CPU Usage in Seconds',
                               "cpu-usage-bp{0}".format(label_suffix))
                # box plot CPU AVG Usage %
                self.__boxplot(avgcpu_bp_df, 'CPU Usage in %',
                               "cpu-percentage-bp{0}".format(label_suffix))

            if container.lower().startswith('memory'):
                mem_bp_df = pandas.DataFrame()
                for run_type, df in data_per_run_dic.items():
                    # source: https://github.com/docker/docker-ce/blob/222348eaf2226f0324a32744ad06d4a7bfe789ac/components/cli/cli/command/container/stats_helpers.go#L225
                    mem_bp_df[run_type] = (df['usage'] - df['stats_cache']).apply(
                        lambda x: x / 1024 / 1024)
                self.__boxplot(mem_bp_df, 'Memory Usage in MiB',
                               "mem-usage-mib-bp{0}".format(label_suffix))

    def get_save_path(self, file_name):
        return os.path.join(self.run_config.evaluation_plots_folder_path, file_name)

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
            ax.plot(x_axis, y, '-', color=self.colorbrewer2_colors(i),
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
        plt.savefig(self.get_save_path(format_filename(label, "pdf")))
        plt.close()

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
            ax.plot(x_axis, y, '-', color=self.colorbrewer2_colors(i),
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
        plt.close()

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
            # source: https://github.com/docker/docker-ce/blob/222348eaf2226f0324a32744ad06d4a7bfe789ac/components/cli/cli/command/container/stats_helpers.go#L225
            y = df['usage'] - df['stats_cache']
            y = y.apply(lambda x: x / 1024 / 1024)
            ax.plot(x_axis, y, '-', color=self.colorbrewer2_colors(i),
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
        plt.close()

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

    def load_provider_csvs(self):
        final_result = {}
        csv_fetch_folder = os.path.join(
            self.run_config.evaluation_csv_folder_path, "provider")
        fetches = [f for f in os.listdir(csv_fetch_folder) if os.path.isfile(
            os.path.join(csv_fetch_folder, f))]
        for csv_file in fetches:
            cat_run = csv_file[:-4]
            load_path = os.path.join(csv_fetch_folder, csv_file)
            final_result[cat_run] = pandas.read_csv(load_path)
        self.provider_metrics = final_result

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
        self.receiver_metrics = final_result

    def __boxplot(self, bp_df, ylabel, filelable):
        bp = bp_df.boxplot()
        bp.set_ylabel(ylabel)
        bp_df.describe().to_csv(self.get_save_path(format_filename(filelable, "csv")))
        plt.savefig(self.get_save_path(format_filename(filelable, "pdf")))
        plt.close()

# format the time according to https://stackoverflow.com/questions/15240003/matplotlib-intelligent-axis-labels-for-timedelta


def format_time_ticks(value, pos):
    "Formats this data to a human readable string"
    timedlt = datetime.timedelta(seconds=value)
    return str(timedlt)


def format_filename(filename, filetype):
    return "{0}.{1}".format(strip_chars(filename), filetype)


def strip_chars(someString):
    """ Provides """
    someString = someString.replace(" ", "-")
    someString = someString.replace("_", "-")
    return "".join(c for c in someString if c in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-")
