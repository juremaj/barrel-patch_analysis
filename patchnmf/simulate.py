import numpy as np

def poiss_train(tau, t_max=10):
    # tau - parameter of exponential distribution
    # t_max - duration of spike train
    
    st = []
    st.append(np.random.exponential(tau)) # first spike time
    count = 0
    
    while st[count] < t_max:
        st.append(st[count] + np.random.exponential(tau)) # subsequent spike times with ISIs from exponential distribution
        count += 1
    
    # removing final spike (outside [0, 10])
    st = st[0:-1]
    
    return np.array(st), count