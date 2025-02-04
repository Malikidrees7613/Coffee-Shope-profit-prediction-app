import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('Coffee shope prediction Model2.h5')

# Function to make prediction (this should be defined somewhere in your code)
def make_prediction(model, input_data):
    # Assume your model expects input data in a specific format and returns predictions
    # Replace this with your actual prediction logic
    predictions = model.predict(input_data)
    return predictions

st.title("Profit Prediction App")

st.write("""
This application predicts the profit for different products based on their unit price, units sold, and total sales.
Enter the values below and click **Predict** to see the forecasted profits.
""")

unit_price = st.number_input('Unit Price', min_value=0.0, step=0.1)
units_sold = st.number_input('Units Sold', min_value=0, step=1)
total_sales = st.number_input('Total Sales', min_value=0.0, step=0.1)

if st.button('Predict'):
    input_data = pd.DataFrame({
        'Unit Price': [unit_price],
        'Units Sold': [units_sold],
        'Total Sales': [total_sales]
    })
    
    # Ensure input_data matches the expected format of your model
    predicted_profit = make_prediction(model, input_data)
    
    st.success(f"Predicted Profit: ${predicted_profit[0][0]:.2f}")

    st.subheader("Visualizations")

    # Scatter Plot of Units Sold vs Predicted Profit
    st.write("### Scatter Plot: Units Sold vs Predicted Profit")
    fig, ax = plt.subplots()
    sns.scatterplot(x=[units_sold], y=predicted_profit.flatten(), ax=ax)
    ax.set_xlabel("Units Sold")
    ax.set_ylabel("Predicted Profit")
    ax.set_title("Units Sold vs Predicted Profit")
    st.pyplot(fig)

    # Distribution Plot of Predicted Profits
    st.write("### Distribution Plot: Predicted Profit")
    fig, ax = plt.subplots()
    sns.histplot(predicted_profit.flatten(), bins=10, kde=True, ax=ax)
    ax.set_xlabel("Predicted Profit")
    ax.set_title("Distribution of Predicted Profits")
    st.pyplot(fig)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    data = input_data.copy()
    data['Predicted Profit'] = predicted_profit.flatten()
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap of Input Features and Predicted Profit")
    st.pyplot(fig)
