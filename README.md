**Microgrid Simulation and Demand Prediction**
=============================================

**Overview**

This project simulates a microgrid with solar, wind, and biogas power generation, as well as electricity demand. It also trains a demand prediction model using a Random Forest Regressor.

**Features**

1. Simulates microgrid power generation and demand over a specified period
2. Trains a demand prediction model using historical data
3. Visualizes microgrid data, including generation and demand plots
4. Saves simulation data to a CSV file (optional)

**Requirements**

``
Python 3.x
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
``

**Model Evaluation**

The demand prediction model is evaluated using the Mean Squared Error (MSE) and R-squared score metrics.

**Data**

The simulation data is generated using the MicroGridSimulation class and includes the following features:
1. Solar power generation
2. Wind power generation
3. Biogas power generation
4. Electricity demand
5. Timestamp
6. Hour of day
7. Day of week
8. Month
9. Lagged demand (previous hour and 24 hours ago)

**Model**

The demand prediction model is a Random Forest Regressor, which is trained using the historical simulation data. The model is evaluated using the Mean Squared Error (MSE) and R-squared score metrics.

**Visualization**

The microgrid data is visualized using Matplotlib and Seaborn, including generation and demand plots.

**License**

This project is licensed under the MIT License.
