Summaray of files uiploaded as of 06/08/2025:

Mean_Reversion.py

-Utilise Moving Average cross overs to initiate trades with given sensitivites
-Using Skopt to optimise moving averge lengths and sensitiies
-Contains class which will optimise statergy in-sample then provide signals (1 for hold asset, 0 for hold cash) for out-of sample for general portfolio algorithms

Monte Carlo Simulations.py

- Creates Discrete Heston model as seen in : Low-bias simulation scheme for the Heston model by
 Inverse Gaussian approximation S. T. TSE and JUSTIN W. L. WAN
- Calibrates model on data via mimising loss function of ( daily log return of true data - model)
- Creates simple example of usage on vanilla put and calls returning option price.

  
