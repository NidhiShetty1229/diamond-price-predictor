"""Diamond price prediction model (Random Forest Regressor).

Reconstructed from the internship report. Run `python model.py` to train the
model on diamonds.csv and save it to diamond_model.pkl.
"""
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(BASE_DIR, "diamonds.csv")
DEFAULT_MODEL = os.path.join(BASE_DIR, "diamond_model.pkl")


class DiamondPricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
        )
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
        self.categorical_columns = ["cut", "color", "clarity"]
        self.numerical_columns = ["carat", "depth", "table", "x", "y", "z"]
        self.is_trained = False
        self.metrics = {}

    # ------------------------------------------------------------------ data
    def load_and_preprocess_data(self, csv_path=DEFAULT_CSV):
        """Load and clean the diamonds dataset."""
        try:
            print(f"Loading dataset from {csv_path}...")
            df = pd.read_csv(csv_path)

            # Drop the unnamed index column if it exists
            if "Unnamed: 0" in df.columns:
                df = df.drop("Unnamed: 0", axis=1)
            elif df.columns[0].startswith("Unnamed"):
                df = df.drop(df.columns[0], axis=1)

            print(f"Original dataset shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")

            required_cols = ["carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            print("Cleaning data...")
            original_size = len(df)

            # Remove rows with zero or negative dimensions
            df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]
            df = df[df["price"] > 0]
            df = df[df["carat"] > 0]

            # Remove extreme outliers
            df = df[df["carat"] <= 5]
            df = df[df["depth"] >= 40]
            df = df[df["depth"] <= 80]
            df = df[df["table"] >= 40]
            df = df[df["table"] <= 80]
            df = df[df["price"] <= 20000]

            # Remove diamonds with inconsistent dimensions
            df["volume"] = df["x"] * df["y"] * df["z"]
            df = df[df["volume"] > 0]

            print(f"Removed {original_size - len(df)} rows during cleaning")
            print(f"Final dataset shape: {df.shape}")
            return df

        except FileNotFoundError:
            print(f"ERROR: Dataset file '{csv_path}' not found.")
            return None
        except Exception as e:
            print(f"ERROR loading dataset: {str(e)}")
            return None

    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features with proper handling of unseen values."""
        df_encoded = df.copy()

        for col in self.categorical_columns:
            if col not in df.columns:
                continue

            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}: {len(le.classes_)} unique values")
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]

                    def safe_transform(value, le=le):
                        if str(value) in le.classes_:
                            return le.transform([str(value)])[0]
                        # Fall back to the first class for unseen values
                        return le.transform([le.classes_[0]])[0]

                    df_encoded[col] = df[col].astype(str).apply(safe_transform)
                else:
                    raise ValueError(f"Label encoder for {col} not found. Model needs to be trained first.")

        return df_encoded

    # -------------------------------------------------------------- training
    def train_model(self, csv_path=DEFAULT_CSV):
        """Train the diamond price prediction model."""
        print("=" * 50)
        print("TRAINING DIAMOND PRICE PREDICTION MODEL")
        print("=" * 50)

        df = self.load_and_preprocess_data(csv_path)
        if df is None:
            print("ERROR: Failed to load dataset")
            return False

        print("\nDataset Statistics:")
        print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        print(f"Carat range: {df['carat'].min():.2f} - {df['carat'].max():.2f}")
        print(f"Cut values: {df['cut'].unique()}")
        print(f"Color values: {df['color'].unique()}")
        print(f"Clarity values: {df['clarity'].unique()}")

        print("\nEncoding categorical features...")
        df_encoded = self.encode_categorical_features(df, fit=True)

        X = df_encoded[self.feature_columns].copy()
        y = df_encoded["price"].copy()
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        print("\nScaling numerical features...")
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[self.numerical_columns] = self.scaler.fit_transform(X_train[self.numerical_columns])
        X_test_scaled[self.numerical_columns] = self.scaler.transform(X_test[self.numerical_columns])

        print("\nTraining Random Forest model...")
        self.model.fit(X_train_scaled, y_train)

        print("Making predictions...")
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

        print("\n" + "=" * 50)
        print("MODEL PERFORMANCE METRICS")
        print("=" * 50)
        print(f"Training MAE: ${train_mae:.2f}")
        print(f"Test MAE: ${test_mae:.2f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test RMSE: ${test_rmse:.2f}")

        feature_importance = pd.DataFrame(
            {"feature": self.feature_columns, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print("\nFeature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")

        print("\nSample Predictions:")
        for i in range(min(5, len(X_test))):
            actual = y_test.iloc[i]
            predicted = y_pred_test[i]
            print(f"Actual: ${actual:.2f}, Predicted: ${predicted:.2f}, Difference: ${abs(actual - predicted):.2f}")

        self.metrics = {
            "train_mae": float(train_mae),
            "test_mae": float(test_mae),
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "test_rmse": test_rmse,
            "training_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "feature_importance": {
                r["feature"]: float(r["importance"]) for _, r in feature_importance.iterrows()
            },
        }

        self.is_trained = True
        print("\n" + "=" * 50)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        return True

    # ------------------------------------------------------------ prediction
    def predict_price(self, diamond_features):
        """Predict diamond price from a dict of features."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Please train the model first.")

        try:
            missing_features = [f for f in self.feature_columns if f not in diamond_features]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

            df = pd.DataFrame([diamond_features])

            for col in self.numerical_columns:
                df[col] = pd.to_numeric(df[col])

            if df["carat"].iloc[0] <= 0:
                raise ValueError("Carat must be positive")
            if any(df[["x", "y", "z"]].iloc[0] <= 0):
                raise ValueError("Dimensions (x, y, z) must be positive")
            if not (40 <= df["depth"].iloc[0] <= 80):
                raise ValueError("Depth should be between 40 and 80")
            if not (40 <= df["table"].iloc[0] <= 80):
                raise ValueError("Table should be between 40 and 80")

            df_encoded = self.encode_categorical_features(df, fit=False)

            df_scaled = df_encoded.copy()
            df_scaled[self.numerical_columns] = self.scaler.transform(df_encoded[self.numerical_columns])

            prediction = self.model.predict(df_scaled[self.feature_columns])
            return max(0.0, float(prediction[0]))  # ensure non-negative

        except Exception as e:
            print(f"ERROR making prediction: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")

    # ----------------------------------------------------------- persistence
    def save_model(self, model_path=DEFAULT_MODEL):
        """Save the trained model and encoders."""
        if not self.is_trained:
            print("WARNING: Saving untrained model")

        model_data = {
            "model": self.model,
            "label_encoders": self.label_encoders,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "is_trained": self.is_trained,
            "metrics": self.metrics,
        }

        try:
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)
            print(f"Model saved successfully to {model_path}")
            return True
        except Exception as e:
            print(f"ERROR saving model: {str(e)}")
            return False

    def load_model(self, model_path=DEFAULT_MODEL):
        """Load a pre-trained model and encoders."""
        try:
            if not os.path.exists(model_path):
                print(f"Model file '{model_path}' not found. Please train the model first.")
                return False

            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.label_encoders = model_data["label_encoders"]
            self.scaler = model_data.get("scaler", StandardScaler())
            self.feature_columns = model_data["feature_columns"]
            self.categorical_columns = model_data["categorical_columns"]
            self.numerical_columns = model_data.get("numerical_columns", self.numerical_columns)
            self.is_trained = model_data.get("is_trained", True)
            self.metrics = model_data.get("metrics", {})

            print(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"ERROR loading model: {str(e)}")
            return False


def main():
    """Train the model, save it and run a few sample predictions."""
    print("Diamond Price Prediction Model")
    print("=" * 40)

    predictor = DiamondPricePredictor()

    if not os.path.exists(DEFAULT_CSV):
        print("ERROR: diamonds.csv not found!")
        print("Please make sure the diamonds.csv file is in the same directory.")
        return

    if predictor.train_model(DEFAULT_CSV):
        predictor.save_model()

        test_diamonds = [
            {"carat": 1.0, "cut": "Ideal", "color": "G", "clarity": "VS2",
             "depth": 61.5, "table": 57.0, "x": 6.5, "y": 6.48, "z": 4.0},
            {"carat": 0.5, "cut": "Premium", "color": "E", "clarity": "VVS1",
             "depth": 62.0, "table": 58.0, "x": 5.1, "y": 5.1, "z": 3.2},
            {"carat": 2.0, "cut": "Good", "color": "J", "clarity": "SI1",
             "depth": 63.0, "table": 59.0, "x": 8.0, "y": 8.0, "z": 5.0},
        ]

        print("\n" + "=" * 50)
        print("SAMPLE PREDICTIONS")
        print("=" * 50)

        for i, diamond in enumerate(test_diamonds, 1):
            try:
                price = predictor.predict_price(diamond)
                print(f"\nDiamond {i}:")
                print(f"Features: {diamond}")
                print(f"Predicted Price: ${price:.2f}")
            except Exception as e:
                print(f"Error predicting diamond {i}: {str(e)}")
    else:
        print("Model training failed!")


if __name__ == "__main__":
    main()
