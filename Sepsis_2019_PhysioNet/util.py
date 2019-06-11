import matplotlib.pyplot as plt
import numpy as np
import torch

def plotter(model_name, utility, p_utility, train_losses, train_pos_acc, train_neg_acc,
            val_losses, val_pos_acc, val_neg_acc, loss_type='Weighted BCE Loss'):
    '''
    Plots loss and accuracy curves
    '''
    plt.subplot(1,2,1)   
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(loss_type)
    plt.title('Loss')
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(train_pos_acc, label='train_pos', c='g', ls='--')
    plt.plot(train_neg_acc, label='train_neg', c='r', ls='--')
    plt.plot(val_pos_acc, label='val_pos', c='g')
    plt.plot(val_neg_acc, label='val_neg', c='r')
    plt.plot(utility, label='utility', c='b', ls='--')
    plt.plot(p_utility, label='p_utility', c='b')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.grid()
    plt.suptitle(model_name)
    plt.show()
    return

    # TODO: combine show_prog and save_prog?

def show_prog(epoch, utility, p_utility, train_loss, val_loss, train_pos_acc, train_neg_acc, 
              val_pos_acc, val_neg_acc, time_elapsed):
    '''
    Prints current epoch's losses, accuracy and runtime
    '''
    print('E %03d --- UTIL: %.3f --- PP UTIL: %.3f --- RUNTIME: %ds' % (epoch+1, utility, p_utility, time_elapsed))
    print('TRAIN  |  loss: %.3f  |  pos acc: %.3f  |  neg acc: %.3f' % (train_loss, train_pos_acc, train_neg_acc))
    print('VALID  |  loss: %.3f  |  pos acc: %.3f  |  neg acc: %.3f' % (val_loss, val_pos_acc, val_neg_acc))
    
def save_prog(model, model_path, utility, p_utility, train_losses, val_losses, train_pos_acc, train_neg_acc,
              val_pos_acc, val_neg_acc, epoch, save_rate):
    '''
    Saves losses and accuracys to model folder
    Saves model state dict every save_rate epochs
    '''
    np.save(model_path +'/train_losses', train_losses)
    np.save(model_path +'/val_losses', val_losses)
    np.save(model_path +'/train_pos_acc', train_pos_acc)
    np.save(model_path +'/train_neg_acc', train_neg_acc)
    np.save(model_path +'/val_pos_acc', val_pos_acc)
    np.save(model_path +'/val_neg_acc', val_neg_acc)
    np.save(model_path +'/utility', utility)
    np.save(model_path +'/p_utility', p_utility)

    if (epoch+1) % save_rate ==0: #save model dict
        torch.save(model.state_dict(), model_path + '/model_epoch%s' % (epoch+1))

def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0, check_errors=True):

    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    return np.sum(u)
        
def evaluate_sepsis_score(labels, prediction):

    dt_early   = -12
    dt_optimal = -6
    dt_late    = 3

    max_u_tp = 1
    min_u_fn = -2
    u_fp     = -0.05
    u_tn     = 0
    num_files= len(labels)
    

    # Compute utility.
    observed_utilities = np.zeros(num_files)
    best_utilities     = np.zeros(num_files)
    worst_utilities    = np.zeros(num_files)
    inaction_utilities = np.zeros(num_files)

    for k in range(num_files):
        num_rows          = len(labels[k])
        observed_predictions = prediction[k]
        best_predictions     = np.zeros(num_rows)
        worst_predictions    = np.zeros(num_rows)
        inaction_predictions = np.zeros(num_rows)

        if np.any(labels[k]):
            t_sepsis = np.argmax(labels[k]) - dt_optimal
            best_predictions[max(0, t_sepsis + dt_early) : min(t_sepsis + dt_late + 1, num_rows)] = 1
        worst_predictions = 1 - best_predictions

        observed_utilities[k] = compute_prediction_utility(labels[k], observed_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        best_utilities[k]     = compute_prediction_utility(labels[k], best_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        worst_utilities[k]    = compute_prediction_utility(labels[k], worst_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        inaction_utilities[k] = compute_prediction_utility(labels[k], inaction_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

    unnormalized_observed_utility = np.sum(observed_utilities)
    unnormalized_best_utility     = np.sum(best_utilities)
    unnormalized_worst_utility    = np.sum(worst_utilities)
    unnormalized_inaction_utility = np.sum(inaction_utilities)

    normalized_observed_utility = (unnormalized_observed_utility - unnormalized_inaction_utility) / (unnormalized_best_utility - unnormalized_inaction_utility)

    return  normalized_observed_utility    