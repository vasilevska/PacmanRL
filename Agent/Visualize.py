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



def counter_plot(counter, name):

    fig = plt.figure()
    plt.bar(counter.keys(), counter.values(), 0.33)

    fig.suptitle(name, fontsize=20)
    plt.xlabel('action', fontsize=18)
    plt.ylabel('freq', fontsize=16)
    
    return fig


def plot_training(epsilons, train_loss, cum_reward, counter):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=[14, 16])


    ax1.plot(epsilons)
    ax1.set_title('Epsilon')
    ax1.set_xlabel('num_of_steps')
    ax1.set_ylabel('epsilon')

    ax2.plot(train_loss)
    ax2.set_title('Train Loss')
    ax2.set_xlabel('num_of_steps')
    ax2.set_ylabel('train_loss')

    ax3.plot(cum_reward)
    ax3.set_title('Cum Reward')
    ax3.set_xlabel('num_of_steps')
    ax3.set_ylabel('cum_reward')

    ax4.bar(counter.keys(), counter.values(), 0.33)
    ax4.set_title('Action Freq')
    ax4.set_xlabel('action')
    ax4.set_ylabel('freq')

    return fig

    
