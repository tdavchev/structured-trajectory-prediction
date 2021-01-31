import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import json

def order(json_file='ohh/results_just_hotel.json'):
    data = []
    with open(json_file, 'r') as f:
        data = json.load(f)
    values = [[] for _ in range(5)]
    counter = 0
    br = 0
    for point in data:
        values[counter].append(point[1][1])
        br += 1
        if br == 9:
            br = 0
            counter += 1

    return np.array(values)

# lstm_values_hotel = order()
#Hotel
rnd_values_hotel = order('ohh/results_rd_hotel.json')
basic_values_hotel = order('ohh/results_just_hotel.json')
social_values_hotel = order('ohh/results_social_hotel.json')
# Univ
rnd_values_univ = order('ohh/results_rd_univ.json')
basic_values_univ = order('ohh/results_just_univ.json')
# old_rnd_values_univ = order('results/univ_results.json')
# basic_values_univ = order('results/just_lstm_results_univ.json')
# new_train_values_univ = order('results/new_train/univ_just_world.json')
# old_values_univ = order('results/univ_just_world.json')
# rnd_values_univ = order('results/rd_univ_results.json')
social_values_univ = order('ohh/results_social_univ.json')
social_values_univ_correct = order('ohh/results_social_univ_correct.json')
# zara01
rnd_values_zara01 = order('ohh/results_rd_zara01.json')
basic_values_zara01 = order('ohh/results_just_zara01.json')
social_values_zara01 = order('ohh/results_social_zara01.json')
# zara02
rnd_values_zara02 = order('ohh/results_rd_zara02.json')
basic_values_zara02 = order('ohh/results_just_zara02.json')
social_values_zara02 = order('ohh/results_social_zara02.json')

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 4)
sns.set_style("dark")
with sns.axes_style("darkgrid"):
    # locs, labels = plt.xticks()
    # plt.xticks(np.arange(9), ('3.2', '4', '4.8', '5.6', '6.4', '7.2', '8', '8.8', '9.6'))
    # means = np.mean(lstm_values_hotel, axis=0)
    # stds = np.std(lstm_values_hotel, axis=0)
    # ax.plot(list(np.arange(9)), means, c=clrs[0], label="R+D")
    # ax.fill_between(list(np.arange(9)), means-stds, means+stds, alpha=0.3, facecolor=clrs[0])

    locs, labels = plt.xticks()
    plt.xticks(np.arange(9), ('3.2', '4', '4.8', '5.6', '6.4', '7.2', '8', '8.8', '9.6'))
    means = np.mean(rnd_values_univ, axis=0)
    stds = np.std(rnd_values_univ, axis=0)
    ax.plot(list(np.arange(9)), means, c=clrs[0], label="RND")
    ax.fill_between(list(np.arange(9)), means-stds, means+stds, alpha=0.3, facecolor=clrs[0])

    locs, labels = plt.xticks()
    plt.xticks(np.arange(9), ('3.2', '4', '4.8', '5.6', '6.4', '7.2', '8', '8.8', '9.6'))
    means = np.mean(basic_values_univ, axis=0)
    stds = np.std(basic_values_univ, axis=0)
    ax.plot(list(np.arange(9)), means, c=clrs[1], label="Just")
    ax.fill_between(list(np.arange(9)), means-stds, means+stds, alpha=0.3, facecolor=clrs[1])

    locs, labels = plt.xticks()
    plt.xticks(np.arange(9), ('3.2', '4', '4.8', '5.6', '6.4', '7.2', '8', '8.8', '9.6'))
    means = np.mean(social_values_univ, axis=0)
    stds = np.std(social_values_univ, axis=0)
    ax.plot(list(np.arange(9)), means, c=clrs[2], label="Social")
    ax.fill_between(list(np.arange(9)), means-stds, means+stds, alpha=0.3, facecolor=clrs[2])


    # locs, labels = plt.xticks()
    # plt.xticks(np.arange(9), ('3.2', '4', '4.8', '5.6', '6.4', '7.2', '8', '8.8', '9.6'))
    # means = np.mean(rnd_values_zara01_bkp, axis=0)
    # stds = np.std(rnd_values_zara01_bkp, axis=0)
    # ax.plot(list(np.arange(9)), means, c=clrs[0], label="RND_bkp")
    # ax.fill_between(list(np.arange(9)), means-stds, means+stds, alpha=0.3, facecolor=clrs[1])

    # locs, labels = plt.xticks()
    # plt.xticks(np.arange(9), ('3.2', '4', '4.8', '5.6', '6.4', '7.2', '8', '8.8', '9.6'))
    # means = np.mean(rnd_values_univ, axis=0)
    # stds = np.std(rnd_values_univ, axis=0)
    # ax.plot(list(np.arange(9)), means, c=clrs[2], label="R+D LSTM glorot action weights")
    # ax.fill_between(list(np.arange(9)), means-stds, means+stds, alpha=0.3, facecolor=clrs[2])

    # locs, labels = plt.xticks()
    # plt.xticks(np.arange(9), ('3.2', '4', '4.8', '5.6', '6.4', '7.2', '8', '8.8', '9.6'))
    # means = np.mean(rnd_values_univ, axis=0)
    # stds = np.std(rnd_values_univ, axis=0)
    # ax.plot(list(np.arange(9)), means, c=clrs[1], label="R+D Univ")
    # ax.fill_between(list(np.arange(9)), means-stds, means+stds, alpha=0.3, facecolor=clrs[1])


    # locs, labels = plt.xticks()
    # plt.xticks(np.arange(9), ('3.2', '4', '4.8', '5.6', '6.4', '7.2', '8', '8.8', '9.6'))
    # means = np.mean(rnd_values_zara01, axis=0)
    # stds = np.std(rnd_values_zara01, axis=0)
    # ax.plot(list(np.arange(9)), means, c=clrs[2], label="R+D Zara01")
    # ax.fill_between(list(np.arange(9)), means-stds, means+stds, alpha=0.3, facecolor=clrs[2])


    # locs, labels = plt.xticks()
    # plt.xticks(np.arange(9), ('3.2', '4', '4.8', '5.6', '6.4', '7.2', '8', '8.8', '9.6'))
    # means = np.mean(rnd_values_zara02, axis=0)
    # stds = np.std(rnd_values_zara02, axis=0)
    # ax.plot(list(np.arange(9)), means, c=clrs[3], label="R+D Zara02")
    # ax.fill_between(list(np.arange(9)), means-stds, means+stds, alpha=0.3, facecolor=clrs[3])

plt.xlabel("Assigned prediction time in seconds")
plt.ylabel("Error")
plt.title("Average Displacement Error over prediction time")
plt.legend()
# plt.axis([3,8,0.005,0.03])
plt.savefig("plotResults_univ.png")
