from flask import Flask, request, render_template
# import numpy as np
# import pandas as pd

from src.pipeline.pred_pipeline import CustomData, PredictPipeline
from src.logger import logging

# from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application


# Route for HomePage
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        logging.info("Getting user input data from web form")
        data = CustomData(
            gender=request.form.get("gender"),
            race=request.form.get("ethnicity"),
            parental_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_prep=request.form.get("test_preparation_course"),
            read_score=request.form.get("reading_score"),
            write_score=request.form.get("writing_score"),
        )
        df_pred = data.get_data_as_frame()
        # print(df_pred)

        logging.info("Getting prediction for user input data")
        pred_pl = PredictPipeline()
        results = pred_pl.predict(features=df_pred)
        logging.info("Returning predicted value to user")
        return render_template("home.html", results=results[0].round(2))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
