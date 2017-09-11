import numpy as np
import matplotlib.pyplot as plt

with np.load('errors.npz') as data:
    history = data['metrics'][()]
    b = data['best_epoch']
    train_acuracy = history['train']['accuracy']
    val_acuracy = history['val']['accuracy']
    train_jaccard = history['train']['jaccard']
    val_jaccard = history['val']['jaccard']
    train_loss = history['train']['loss']
    val_loss = history['val']['loss']

    print(history['train']['loss'][b])
    print(history['train']['accuracy'][b])
    print(history['train']['jaccard'][b])

    print(history['val']['loss'][b])
    print(history['val']['accuracy'][b])
    print(history['val']['jaccard'][b])

    curr_ep = len(history['train']['loss'])
    print(b)
    print(curr_ep)

    epochs = np.arange(curr_ep)

    # acuracy plot =====================
    y_stack = np.row_stack((np.array(train_acuracy), np.array(val_acuracy)))

    fig = plt.figure(figsize=(25, 8))
    ax1 = fig.add_subplot(111)

    ax1.plot(epochs, y_stack[0, :], label='train', color='c', marker='o')
    ax1.plot(epochs, y_stack[1, :], label='val', color='g', marker='o')
    ax1.legend(loc=2)

    plt.xticks(epochs)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    handles, labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(.1, 1))

    plt.savefig('acuracy.png')

    # loss plot =====================
    y_stack = np.row_stack((np.array(train_loss), np.array(val_loss)))

    fig = plt.figure(figsize=(25, 8))
    ax1 = fig.add_subplot(111)

    ax1.plot(epochs, y_stack[0, :], label='train', color='c', marker='o')
    ax1.plot(epochs, y_stack[1, :], label='val', color='g', marker='o')
    ax1.legend(loc=2)

    plt.xticks(epochs)
    plt.xlabel('epochs')
    plt.ylabel('loss')

    handles, labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(.1, 1))

    plt.savefig('loss.png')

    # jaccard plot =====================
    y_stack = np.row_stack((np.array(train_jaccard), np.array(val_jaccard)))

    fig = plt.figure(figsize=(25, 8))
    ax1 = fig.add_subplot(111)

    ax1.plot(epochs, y_stack[0, :], label='train', color='c', marker='o')
    ax1.plot(epochs, y_stack[1, :], label='val', color='g', marker='o')
    ax1.legend(loc=2)

    plt.xticks(epochs)
    plt.xlabel('epochs')
    plt.ylabel('mean jaccard')

    handles, labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(.1, 1))

    plt.savefig('jaccard.png')