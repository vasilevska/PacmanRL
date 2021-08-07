import numpy as np
import matplotlib.pyplot as plt



def smooth_result(result, factor=10):
    result = np.array(result)
    size = int(result.shape[0] / factor)
    result = np.convolve(result, np.ones(size) / size, mode='valid')
    return result


def visualize_result(returns, td_errors, policy_errors=None):

    if policy_errors is None:
        fig, (ax1, ax2) = plt.subplots(2, figsize=[6.4, 6], gridspec_kw={'height_ratios': [2, 1]})
        ax3 = None
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=[6.4, 8], gridspec_kw={'height_ratios': [2, 1, 1]})

    ax1.plot(returns, label='returns', color='#3366ff')
    ax1.plot(smooth_result(returns), label='smoothed', color='#000066')
    ax1.set_title('Returns')
    ax1.set_xlabel('episodes')
    ax1.legend()

    ax2.plot(td_errors, color='#ff3300')
    ax2.set_xlabel('update steps')
    ax2.set_title('TD Errors')

    if policy_errors is not None:
        ax3.plot(policy_errors, color='#ff0000')
        ax3.set_xlabel('update steps')
        ax3.set_title('Policy Loss')

    fig.tight_layout()
    return fig