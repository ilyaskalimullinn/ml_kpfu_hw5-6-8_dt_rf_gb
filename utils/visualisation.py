import os.path

import pandas as pd
from plotly import express as px


class Visualisation:

    def __init__(self, graphs_dir):
        self.graphs_dir = graphs_dir

    def visualize_best_models(self, experiments, rounding=3, title='Лучшие модели Random Forest', file_name=None):
        df = pd.DataFrame({
            "L_1": [round(exp["L_1"], rounding) for exp in experiments],
            "L_2": [round(exp["L_2"], rounding) for exp in experiments],
            "M": [round(exp["M"], rounding) for exp in experiments],
            "accuracy_valid": [round(exp["accuracy_valid"], rounding) for exp in experiments],
            "accuracy_test": [round(exp["accuracy_test"], rounding) for exp in experiments]
        })

        df["model_params"] = "L_1=" + df["L_1"].astype(str) + ", L_2=" + df["L_2"].astype(str) + ", M=" +\
                             df["M"].astype(str)

        fig = px.line(data_frame=df,
                         x='model_params',
                         y='accuracy_valid',
                         hover_data={'L_1': True,
                                     'L_2': True,
                                     'M': True,
                                     "accuracy_valid": True,
                                     "accuracy_test": True,
                                     'model_params': False
                                     })

        fig.update_layout(
            title=title,
            xaxis_title="Параметры модели",
            yaxis_title="Ошибка на валидационной выборке",
            legend_title="Legend",
            font_size=14
        )

        fig.show()

        if file_name:
            fig.write_html(os.path.join(self.graphs_dir, file_name))

    def visualize_confusion_matrix(self, confusion_matrix, title='Confusion matrix', file_name=None):
        fig = px.imshow(confusion_matrix, text_auto=True)

        fig.update_layout(
            xaxis_title="Фактический класс",
            yaxis_title="Предсказанный класс",
            title=title,
            legend_title="Legend",
            font_size=14
        )

        fig.show()

        if file_name:
            fig.write_html(os.path.join(self.graphs_dir, file_name))
