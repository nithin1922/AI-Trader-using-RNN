# AI Trader

## Overview
AI Trader is a deep reinforcement learning-based trading bot that uses a neural network to make stock trading decisions. It aims to maximize profits by deciding whether to buy, hold, or sell stocks based on past price data.

## Features
- Uses a deep neural network for stock trading decisions
- Implements reinforcement learning with an epsilon-greedy policy
- Utilizes past stock price data for predictions
- Supports training over multiple episodes
- Saves trained models periodically

## Requirements
The project requires the following Python libraries:

- TensorFlow
- NumPy
- Pandas
- Pandas DataReader
- Matplotlib
- tqdm

## Installation
To install the necessary dependencies, run:
```bash
pip install tensorflow numpy pandas pandas-datareader matplotlib tqdm
```

## Usage
1. Load stock data using `data_reader.DataReader("AAPL", data_source="yahoo")`.
2. Initialize the AI Trader with:
   ```python
   trader = AI_TRADER(window_size=10)
   ```
3. Train the model over multiple episodes:
   ```python
   for episode in range(1, episodes + 1):
       ...  # Training loop
   ```
4. Save the trained model periodically.
5. Use the trained model for making trade decisions.

## Model Architecture
The neural network consists of:
- Input layer with `state_size` features
- Hidden layers with 32, 64, and 128 neurons using ReLU activation
- Output layer with 3 neurons representing HOLD, BUY, and SELL actions

## Training
- Uses experience replay with a memory buffer of size 2000
- Trains in mini-batches of size 32
- Uses an Adam optimizer with a learning rate of 0.001
- Implements an epsilon-greedy strategy for action selection

## Trading Strategy
1. **BUY**: If AI Trader predicts an upward trend.
2. **SELL**: If AI Trader predicts a downward trend and has stocks in inventory.
3. **HOLD**: If no favorable trade is detected.

## Results
The AI Trader prints the total profit at the end of each episode and saves models every 10 episodes.

## Future Improvements
- Implement LSTM-based predictions for better trend analysis
- Fine-tune hyperparameters for improved performance
- Add real-time trading functionality with API integration

## License
This project is licensed under the MIT License.

