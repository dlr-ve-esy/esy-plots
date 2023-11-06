from bokeh.plotting import figure


class Line:


    def __init__(
            self, data,
            title="", x_axis_label="", y_axis_label="",
            legend_label="", figure_kwargs=None, line_kwargs=None,
        ) -> None:
            self.data = data

            self.title = title
            self.x_axis_label = x_axis_label
            self.y_axis_label = y_axis_label
            self.legend_label = legend_label
            self.figure_kwargs = figure_kwargs if figure_kwargs is not None else {}
            self.line_kwargs = line_kwargs if line_kwargs is not None else {}

    def __call__(self, option):
        fig = figure(
            title=self.title,
            x_axis_label=self.x_axis_label,
            y_axis_label=self.y_axis_label
        )

        x, y = self._slice(option)

        fig.line(x, y, legend_label=self.legend_label, line_width=2)
        return fig

    def _slice(self, option):
        x = self.data.index
        y = self.data[option].values
        return x, y