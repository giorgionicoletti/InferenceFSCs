- Merge the generation / inference class with seamless switching between numpy and torch (we need numpy for numba compatibility in the generation part), or create a superclass that handles the switching

- Implement custom actions / observation spaces, to be handled by dictionaries so that actions can be passed as strings, if needed; same with observations in the discrete case

- Use the full KL divergence for the empirical probability as a metric during trainig, even if it's not used for the loss function

- Check again the description of each function and class, and add more details if needed; fix with the recent updates