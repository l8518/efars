import datetime
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas
from matplotlib.ticker import FuncFormatter
from pandas.plotting import register_matplotlib_converters
import numpy
import seaborn as sns

register_matplotlib_converters()

plt.rcParams['figure.figsize'] = (7.0, 6.0)


class Plotter():
    def __init__(self, run_config):
        sns.set()
        sns.set(font_scale=1.2)
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
        plotdf = rdf

        x_axis = rdf.index

        # define axis
        f1_axis = plotdf['f1 score']
        receiver_errors_axis = plotdf['failure_indice']
        provider_errors_axis = plotdf['provisions_failed_cnt']

        # plot f1 metrics and failed fetches in one diagram:
        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(111)

        self.plot_f1_with_failures(
            fig, ax1, x_axis, f1_axis, provider_errors_axis, receiver_errors_axis, f1_score_max, failure_max, duration_max)

        self.__save_plot("f1-failure-{0}".format(label_suffix), plotdf)

    def plot_f1_with_failures(self, fig, ax1, x_axis, f1_axis, provider_f_axis, receiver_f_axis, f1_score_max, failure_max, duration_max):
        ax2 = ax1.twinx()

        # plot f1 metrics and failed fetches in one diagram:
        ax2.plot(x_axis, f1_axis, '-', zorder=10,
                 linewidth=2, color='r', label='F1 Score')
        ax2.yaxis.tick_left()
        ax2.set_ylabel('F1 Score')

        ax2.set_ylim(bottom=0, top=f1_score_max)
        ax2.yaxis.set_label_position("left")

        formatter = matplotlib.ticker.FuncFormatter(format_time_ticks)
        ax1.xaxis.set_major_formatter(formatter)
        ax1.set_xlim(left=max(x_axis)*-0.02, right=max(x_axis)*1.02)

        # plot the failures as bars
        bar_width = (len(x_axis) * 5)
        ax1.set_ylim(bottom=0, top=failure_max)
        ax1.bar(x_axis, receiver_f_axis, bar_width,
                edgecolor='none', zorder=1, label='Failed Requests')
        ax1.bar(x_axis, provider_f_axis, width=bar_width*0.4,
                edgecolor='none', zorder=2, label='Failed Provisions')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.set_ylabel('Average Failures')
        ax1.set_xlabel('Duration (h:mm:ss)')
        ax1.grid(False)
        ax1.grid(which='major', axis='x', linestyle='--')

        # ask matplotlib for the plotted objects and their labels
        fig.legend(loc=4)
        fig.subplots_adjust(bottom=0.2)

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
        self.__fitted_boxplot(bp_f1_df, 'F1 Score', "f1-bp")
        self.__fitted_boxplot(bp_receiver_fail_df,
                              'Average Failed Requests', "receiver-failure-bp")
        self.__fitted_boxplot(bp_provider_fail_df,
                              'Average Failed Provisions', "provider-failure-bp")

    def calculate_performance_results(self):
        # CPU usage is measured in "user jiffies (1/100 th of a second)" (aka 10ms)
        for container, data_per_run_dic in self.monitor_data.items():
            label_suffix = "{0}".format(container)
            if container.lower().startswith('cpu'):
                self.cpu_usage_per_run(data_per_run_dic, label_suffix)
                self.cpu_usage_percent_per_run(data_per_run_dic, label_suffix)
            if container.lower().startswith('memory'):
                self.mem_usage_per_run(data_per_run_dic, label_suffix)
            if container.lower().startswith('block io'):
                self.blkio_per_run(data_per_run_dic, label_suffix, "bytes_write_sum",
                                   "Block Storage Written in GiB", "blkio-written")
                self.blkio_per_run(data_per_run_dic, label_suffix,
                                   "bytes_read_sum", "Block Storage Read in GiB", "blkio-read")

    def calculate_performance_boxplots(self):
        # CPU usage is measured in "user jiffies (1/100 th of a second)" (aka 10ms)
        grouped_cpu_usage = pandas.DataFrame()
        grouped_cpu_perc = pandas.DataFrame()
        for container, data_per_run_dic in self.monitor_data.items():
            label_suffix = "{0}".format(container)
            if container.lower().startswith('cpu'):
                cpu_bp_df = pandas.DataFrame()
                avgcpu_bp_df = pandas.DataFrame()
                for run_type, df in data_per_run_dic.items():
                    cpu_bp_df[run_type] = (
                        df['cpu_stats_cpu_usage_total_usage'])
                    avgcpu_bp_df[run_type] = (df['avg_cpu'])
                # box plot CPU Usage in Seconds
                self.__fitted_boxplot(cpu_bp_df, 'CPU Usage in Seconds',
                                      "cpu-usage-bp-{0}".format(label_suffix))
                # box plot CPU AVG Usage %
                self.__fitted_boxplot(avgcpu_bp_df, 'CPU Usage in %',
                                      "cpu-percentage-bp-{0}".format(label_suffix), yax_formatter=FuncFormatter('{0:0.1%}'.format))
                cpu_bp_df["Component"] = container.lstrip("CPU ")
                grouped_cpu_usage = grouped_cpu_usage.append(cpu_bp_df)
                avgcpu_bp_df["Component"] = container.lstrip("CPU ")
                grouped_cpu_perc = grouped_cpu_perc.append(avgcpu_bp_df)
        # Plot Grouped
        self.__boxplot(grouped_cpu_usage, 'CPU Usage in Seconds',
                       "cpu-usage-bp-grouped", grp_by="Component")
        self.__boxplot(grouped_cpu_perc, 'CPU Usage in %',
                       "cpu-percent-bp-grouped", grp_by="Component",
                       yax_formatter=FuncFormatter('{0:0.1%}'.format))

        grouped_mem = pandas.DataFrame()
        for container, data_per_run_dic in self.monitor_data.items():
            if container.lower().startswith('memory'):
                mem_bp_df = pandas.DataFrame()
                for run_type, df in data_per_run_dic.items():
                    # source: https://github.com/docker/docker-ce/blob/222348eaf2226f0324a32744ad06d4a7bfe789ac/components/cli/cli/command/container/stats_helpers.go#L225
                    mem_bp_df[run_type] = (df['usage'] - df['stats_cache']).apply(
                        lambda x: x / 1024 / 1024)
                self.__fitted_boxplot(mem_bp_df, 'Memory Usage in MiB',
                                      "mem-usage-mib-bp-{0}".format(label_suffix))
                mem_bp_df["Component"] = container.lstrip("Memory ")
                grouped_mem = grouped_mem.append(mem_bp_df)
        self.__boxplot(grouped_mem, 'Memory Usage in MiB',
                       "mem-usage-mib-bp-grouped", grp_by="Component")

        grouped_bw = pandas.DataFrame()
        grouped_br = pandas.DataFrame()
        for container, data_per_run_dic in self.monitor_data.items():
            if container.lower().startswith('block io'):
                bwby_df = pandas.DataFrame()
                brby_df = pandas.DataFrame()
                for run_type, df in data_per_run_dic.items():
                    bwby_df[run_type] = (df['bytes_write_sum']).apply(
                        lambda x: x / (1024 ** 3))
                    brby_df[run_type] = (df['bytes_read_sum']).apply(
                        lambda x: x / (1024 ** 3))
                self.__fitted_boxplot(bwby_df, 'Block Storage Written in GiB',
                                      "blkio-written-bp-{0}".format(label_suffix))
                self.__fitted_boxplot(brby_df, 'Block Storage Read in GiB',
                                      "blkio-read-bp-{0}".format(label_suffix))
                bwby_df["Component"] = container.replace("Block IO ", "")
                grouped_bw = grouped_bw.append(bwby_df)
                brby_df["Component"] = container.replace("Block IO ", "")
                grouped_br = grouped_br.append(brby_df)
        self.__boxplot(grouped_bw, 'Block Storage Written in GiB',
                       "blkio-written-bp-grouped", grp_by="Component")
        self.__boxplot(grouped_br, 'Block Storage Read in GiB',
                       "blkio-read-bp-grouped", grp_by="Component")

    def get_save_path(self, file_name):
        return os.path.join(self.run_config.evaluation_plots_folder_path, file_name)

    def cpu_usage_percent_per_run(self, data_per_run_dic, label_suffix):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        #fig, ax = plt.subplots(1, 1)
        fig.suptitle(label_suffix[4:])
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
            ax.plot(x_axis, y, '-',
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
        plt.autoscale()
        self.__save_plot(
            "cpu-usage-percent-{0}".format(label_suffix), *data_per_run_dic.values())

    def cpu_usage_per_run(self, data_per_run_dic, label_suffix):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        fig.suptitle(label_suffix[4:])
        ax.set_ylabel('CPU Time in 1/100 th Seconds')
        ax.set_xlabel('Duration (h:mm:ss)')

        x_max = 0
        y_max = 0
        i = -1
        for run_type, df in data_per_run_dic.items():
            i += 1
            df = df.set_index("time_elapsed")
            x_axis = df.index
            # convert to jiffies to seconds, according to https://docs.docker.com/v17.09/engine/admin/runmetrics/#cpu-metrics-cpuacctstat
            y = df['cpu_stats_cpu_usage_total_usage']
            ax.plot(x_axis, y, '-',
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
        plt.autoscale()
        self.__save_plot(
            "cpu-usage-{0}".format(label_suffix), *data_per_run_dic.values())

    def mem_usage_per_run(self, data_per_run_dic, label_suffix):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        fig.suptitle(label_suffix[7:])
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
            ax.plot(x_axis, y, '-',
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
        self.__save_plot(
            "mem-usage-{0}".format(label_suffix), *data_per_run_dic.values())

    def blkio_per_run(self, data_per_run_dic, label_suffix, blkio_col, ylabel, file_prefix):
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(label_suffix[9:])
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Duration (h:mm:ss)')

        x_max = 0
        y_max = 0
        i = -1
        for run_type, df in data_per_run_dic.items():
            i += 1
            df = df.set_index("time_elapsed")
            x_axis = df.index
            # source: https://github.com/docker/docker-ce/blob/222348eaf2226f0324a32744ad06d4a7bfe789ac/components/cli/cli/command/container/stats_helpers.go#L225
            y = df[blkio_col]
            y = y.apply(lambda x: x / (1024 ** 3))
            ax.plot(x_axis, y, '-',
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
        self.__save_plot(
            "{0}-{1}".format(file_prefix, label_suffix), *data_per_run_dic.values())

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

    def __save_plot(self, file_label, *dfs, grp_by=None):
        file_label = file_label.lower()
        for idx, df in enumerate(dfs):
            desc_file_label = file_label
            if len(dfs) > 1:
                desc_file_label = "{0}_{1}".format(file_label, idx)
            desc = df
            if grp_by is not None:
                desc = df.groupby(grp_by)
            desc.describe().to_csv(self.get_save_path(format_filename(desc_file_label, "csv")))
        plt.savefig(self.get_save_path(format_filename(file_label, "pdf")))
        plt.close()

    def __boxplot(self, bp_df, ylabel, file_label, figsize=None, grp_by=None, yax_formatter=None):
        # Support Newline for "spaced" Names
        if grp_by is not None:
            bp_df[grp_by] = list(map(lambda x: x.replace(" ", "\n"), bp_df[grp_by]))
        bp_df.columns = list(map(lambda x: x.replace(" ", "\n"), bp_df.columns))
        axes = None
        if grp_by is not None:
            axes = bp_df.boxplot(by=grp_by, figsize=figsize)
        else:
            axes = bp_df.boxplot(figsize=figsize)
        # assigned as list, to allow iteration
        # (because grouped boxplox with more then one group
        # return axes as nparray)
        if not isinstance(axes, (list, numpy.ndarray)):
            axes = [axes]

        if isinstance(axes, (numpy.ndarray)):
            axes = axes.flatten()
        for single_axes in axes:
            if (yax_formatter) is not None:
                single_axes.yaxis.set_major_formatter(yax_formatter)
            single_axes.set_ylabel(ylabel)
            single_axes.get_figure().tight_layout()
            single_axes.set_xlabel("")  # remove grp_by label set by pandas
        plt.subplots_adjust()
        plt.tight_layout()
        plt.suptitle("")  # remove suptitle set by pandas
        self.__save_plot(file_label, bp_df, grp_by=grp_by)

    def __fitted_boxplot(self, bp_df, ylabel, file_label, grp_by=None, yax_formatter=None):
        figsize = self.__boxplot_figsize(bp_df)
        return self.__boxplot(bp_df, ylabel, file_label, figsize=figsize, grp_by=grp_by, yax_formatter=yax_formatter)

    def __boxplot_figsize(self, df):
        return (2 * len(df.columns), 6)


def format_time_ticks(value, pos):
    "Formats this data to a human readable string"
    # format the time according to https://stackoverflow.com/questions/15240003/matplotlib-intelligent-axis-labels-for-timedelta
    timedlt = datetime.timedelta(seconds=value)
    hours, remainder = divmod(timedlt.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{h:02.0f}:{m:02.0f}:{s:02.0f}".format(h=hours, m=minutes, s=seconds)


def format_filename(filename, filetype):
    return "{0}.{1}".format(strip_chars(filename), filetype)


def strip_chars(someString):
    """ Provides """
    someString = someString.replace(" ", "-")
    someString = someString.replace("_", "-")
    return "".join(c for c in someString if c in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-")
