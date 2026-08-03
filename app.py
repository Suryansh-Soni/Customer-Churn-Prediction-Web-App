# coding: utf-8

import os
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset and model
df_1 = pd.read_csv(os.path.join(BASE_DIR, "first_telc.csv"))

with open(os.path.join(BASE_DIR, "model.sav"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "columns.pkl"), "rb") as f:
    model_columns = pickle.load(f)


@app.route("/")
def loadPage():
    return render_template("home.html", query="")


@app.route("/", methods=["POST"])
def predict():
    try:
        inputQuery1 = int(request.form["query1"])
        inputQuery2 = float(request.form["query2"])
        inputQuery3 = float(request.form["query3"])

        inputQuery4 = request.form["query4"]
        inputQuery5 = request.form["query5"]
        inputQuery6 = request.form["query6"]
        inputQuery7 = request.form["query7"]
        inputQuery8 = request.form["query8"]
        inputQuery9 = request.form["query9"]
        inputQuery10 = request.form["query10"]
        inputQuery11 = request.form["query11"]
        inputQuery12 = request.form["query12"]
        inputQuery13 = request.form["query13"]
        inputQuery14 = request.form["query14"]
        inputQuery15 = request.form["query15"]
        inputQuery16 = request.form["query16"]
        inputQuery17 = request.form["query17"]
        inputQuery18 = request.form["query18"]

        inputQuery19 = int(request.form["query19"])

        data = [[
            inputQuery1,
            inputQuery2,
            inputQuery3,
            inputQuery4,
            inputQuery5,
            inputQuery6,
            inputQuery7,
            inputQuery8,
            inputQuery9,
            inputQuery10,
            inputQuery11,
            inputQuery12,
            inputQuery13,
            inputQuery14,
            inputQuery15,
            inputQuery16,
            inputQuery17,
            inputQuery18,
            inputQuery19
        ]]

        new_df = pd.DataFrame(
            data,
            columns=[
                "SeniorCitizen",
                "MonthlyCharges",
                "TotalCharges",
                "gender",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "tenure",
            ],
        )

        df_2 = pd.concat([df_1, new_df], ignore_index=True)

        # Convert TotalCharges to numeric
        df_2["TotalCharges"] = pd.to_numeric(
            df_2["TotalCharges"], errors="coerce"
        )

        # Fill missing values
        df_2["TotalCharges"] = df_2["TotalCharges"].fillna(
            df_2["TotalCharges"].median()
        )

        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

        df_2["tenure_group"] = pd.cut(
            df_2["tenure"].astype(int),
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=labels,
            include_lowest=True,
        )

        # Drop tenure column
        df_2.drop(columns=["tenure"], inplace=True)

        new_df_dummies = pd.get_dummies(
            df_2[
                [
                    "gender",
                    "SeniorCitizen",
                    "Partner",
                    "Dependents",
                    "PhoneService",
                    "MultipleLines",
                    "InternetService",
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                    "Contract",
                    "PaperlessBilling",
                    "PaymentMethod",
                    "tenure_group",
                ]
            ]
        )

        # Match training columns
        new_df_dummies = new_df_dummies.reindex(
            columns=model_columns,
            fill_value=0,
        )

        single = model.predict(new_df_dummies.tail(1))
        probability = model.predict_proba(new_df_dummies.tail(1))[:, 1]

        if single[0] == 1:
            o1 = "This customer is likely to be churned!!"
        else:
            o1 = "This customer is likely to continue!!"

        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)

        return render_template(
            "home.html",
            output1=o1,
            output2=o2,
            **request.form,
        )

    except Exception as e:
        print("ERROR:", e)
        return f"<h2>Internal Server Error</h2><pre>{str(e)}</pre>", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)