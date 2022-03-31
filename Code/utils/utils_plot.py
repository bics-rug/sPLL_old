import os
from brian2 import *

def plot_spikes_from_TextureRecognition_structure(Rate_nWTA, Fake_spikes, Osc_spikes, nWTA_spikes, neurons_n,
                                                  save_folder = [], save = False, plot = False):
    # Create figure structure
    # if plot == False:
    #     matplotlib.use('Agg')
    fig1, axis1 = plt.subplots(nrows=3, ncols=int(np.ceil(neurons_n / 2)), sharex=True, sharey=True)
    fig1.suptitle('Rate(Hz) of Input vs PLL')

    gs = axis1[2, 1].get_gridspec()
    for ax in axis1[2, :]:
        ax.remove()
    axbig = fig1.add_subplot(gs[2, :])
    axbig.set_title('nWTA')
    axbig.plot(nWTA_spikes.t, nWTA_spikes.i, 'o')
    winner, failures_here = find_winner(Rate_nWTA, allow_half_winners=True)
    axbig.plot([winner[0], winner[0]], color='b')

    ## Plot the rates of the different PLL vs the input rate
    row = 0
    column = 0
    for neu in range(neurons_n):
        list_here = [neu]

        # fig2.suptitle('WTA Freq(Hz)')

        aaa = Isi_calculator(spikes=Fake_spikes, neuron_list=[0], axis_point=axis1[row, column])
        aaa = Isi_calculator(spikes=Osc_spikes, neuron_list=list_here, axis_point=axis1[row, column])
        # aaa = Isi_calculator(spikes=nWTA_spikes, neuron_list=list_here, axis_point=axis1[row, column])
        if column == int(np.ceil(20 / 2)) - 1:
            row += 1
            column = 0
        else:
            column += 1

    if save == True:
        fig1.savefig(save_folder + "spikes.png")
    if plot == False:
        plt.close(fig1)


def Plot_VoltageSignal_sPLL(neu = [1, 5, 9],interval = [5.2,5.3]):
    aaa = Ref_values.Ie_h
    iii = Ref_values.Ie_NMDA_h
    uuu = Ref_values.v
    ooo = aaa[1]
    interval = [0.5, 0.7]
    # my_sampling = 1000*us
    eee, eee1 = plt.subplots(nrows=6, ncols=1, sharey=True, sharex=True)
    color_list = [(122 / 255, 122 / 255, 1), (204 / 255, 0, 15 / 255), (127 / 255, 190 / 255, 0)]
    for i in range(3):
        eee1[i * 2].plot([interval[0] * second + i * my_sampling for i in
                          range(int((interval[1] - interval[0]) * second / my_sampling))],
                         aaa[i][int(interval[0] * second / my_sampling):int(
                             interval[1] * second / my_sampling) - 1] / aaa.max())
        eee1[i * 2].plot([interval[0] * second + i * my_sampling for i in
                          range(int((interval[1] - interval[0]) * second / my_sampling))],
                         iii[i][int(interval[0] * second / my_sampling):int(
                             interval[1] * second / my_sampling) - 1] / iii.max())
        eee1[i * 2 + 1].plot([interval[0] * second + i * my_sampling for i in
                              range(int((interval[1] - interval[0]) * second / my_sampling))],
                             uuu[i][int(interval[0] * second / my_sampling):int(
                                 interval[1] * second / my_sampling) - 1] / uuu.max())
        selection = np.where(Ref_spikes.i == neu[i])[0]
        eee1[i * 2].plot([Fake_spikes.t, Fake_spikes.t], [0, 1], '--', color='y')
        eee1[i * 2].plot(Ref_spikes.t[selection], - neu[i] + Ref_spikes.i[selection], '.', color=color_list[i])
        selection = np.where(TDE_spikes.i == neu[i])[0]
        eee1[i * 2].plot(TDE_spikes.t[selection], - neu[i] + TDE_spikes.i[selection] + 1, '.', color=color_list[i])
        eee1[1].set_xlim(interval[0], interval[1])
        legend(['FAC', 'TDE_IN', 'Vmem'])


def Plot_SpikeFreq_vs_Texture(rate_in,simulation_array):
    test = np.array([simulation_array[i][1][3] for i in range(len(simulation_array))])
    unici = np.unique(test)
    average_freq = []
    std_freq = []
    for width in unici:
        aaa = np.where(test == width)
        average_freq.append(np.mean(rate_in[0][aaa]))
        std_freq.append(np.std(rate_in[0][aaa]))

    plot([simulation_array[i][1][3] for i in range(len(simulation_array))], rate_in[0], 'o', color='slateblue')
    plot(unici, average_freq, color='navy')
    plt.fill_between(unici, [average_freq[i] - std_freq[i] for i in range(len(unici))],
                     [average_freq[i] + std_freq[i] for i in range(len(unici))], alpha=0.5, color='lavender')

    xlabel('Space distance Texture(px)')
    ylabel('Spike Frequency')
    grid()


def Reduce_ticks_labels(my_ticks,ticks_steps = 10):
    new_ticks = []
    counter = 0
    for label in my_ticks:
        if mod(counter,ticks_steps) == 0:
            new_ticks.append(label)
        else:
            new_ticks.append('')
        counter += 1
    return new_ticks

def plot_confusion_matrix(confusion_matrix, parameters, ticks_steps = 1, force_ticks = None, save = False, save_folder = None):
    if force_ticks == None:
        my_yticks = [i for i in range(confusion_matrix.shape[0])]
        my_yticks = Reduce_ticks_labels(my_ticks=my_yticks, ticks_steps=ticks_steps)
    else:
        my_yticks = [i for i in range(confusion_matrix.shape[0])]
        ticks_minimum = np.min(force_ticks)
        ticks_maximum = np.max(force_ticks)
        ticks_interval = (ticks_maximum - ticks_minimum) / len(my_yticks)
        yticks_labels = list(ticks_minimum + np.array(my_yticks) * ticks_interval)
        yticks_labels = Reduce_ticks_labels(my_ticks=yticks_labels, ticks_steps=ticks_steps)

    xticks = [i for i in range(confusion_matrix.shape[1])]
    f1 = plt.subplot()
    im = f1.imshow(confusion_matrix, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
    plt.yticks(my_yticks, tuple(list(map(str, yticks_labels))))
    xticks_label = list(map(str, xticks[:-1])) + ['NaN']
    plt.xticks(xticks, tuple(xticks_label))
    plt.xlabel('WTA (# Neuron)')
    plt.ylabel('Input(Hz)')
    plt.title('Confusion Matrix')
    plt.colorbar(im)
    if save == True:
        plt.savefig(os.path.join(save_folder, 'confusion_matrix.png'))
        np.save(os.path.join(save_folder, "Confusion_matrix.npy"), confusion_matrix)
        np.save(os.path.join(save_folder, "Freq_Intervals.npy"), parameters['frequencies'])

