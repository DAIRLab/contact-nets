import pdb  # noqa

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend('pgf')


def generate_figure(regenerate_data):
    '''
    TRAINING PARAMETERS
    '''
    # dataset sizes
    N = 20  # training & test dataset size
    N_batch = 1  # batch size
    N_batches = int(N / N_batch)  # number of batches

    # network params
    H = 128  # hidden units

    # optimizer parameters
    learning_rate = 1e-4
    N_epoch = 10000  # max epochs
    estop = 50  # early stopping patience in epochs



    '''
    PHYSICS/SYSTEM PARAMETERS
    '''
    # gravity, initial condition height, timestep, true ground height
    g, q0, dt, GH = 9.81, 1, 1, 0

    # predictor of final state (q'v')
    def nominalModel(v0, gh):
        qF = q0 + v0 * dt - 0.5 * g * (dt ** 2)
        vF = v0 - (g * dt)
        vF[qF < gh] = 0.0
        qF[qF < gh] = gh
        return torch.cat((qF, vF), dim=1)



    '''
    MODEL CONSTRUCTION
    '''
    # END-TO-END MODEL
    Nonlinearity = torch.nn.Tanh
    modelF = torch.nn.Sequential(
        torch.nn.Linear(1, H),
        Nonlinearity(),
        torch.nn.Linear(H, H),
        Nonlinearity(),
        torch.nn.Linear(H, H),
        Nonlinearity(),
        torch.nn.Linear(H, 2),
    )
    lossF = torch.nn.MSELoss(reduction='mean')


    class GroundHeightQP(torch.nn.Module):
        '''GROUND-HEIGHT MODEL WITH QP LOSS.'''
        def __init__(self, gh):
            super().__init__()
            self.gh = torch.nn.Parameter(gh, requires_grad=True)

        def forward(self, data):
            if self.training:
                return self.gh
            else:
                return nominalModel(data, self.gh)

    modelGH = GroundHeightQP(torch.randn(1)[0] * 0.3)


    MSELOSS = torch.nn.MSELoss(reduction='mean')
    RELU = torch.nn.ReLU()

    def lossGH(gh, d):
        v0_l = d[:, 0]
        vf_l = d[:, 2]
        qf_l = d[:, 1]

        # extract net contact impulse
        v_nc = v0_l - (g * dt)
        F = vf_l - v_nc

        #
        height_above_ground = qf_l - gh

        # optimal_impulse = min_impulse (height_above_ground * impulse)^2 + (F - impulse)^2
        optimal_impulse = F / (1 + height_above_ground ** 2)
        optimal_impulse = RELU(optimal_impulse)

        return MSELOSS(height_above_ground * optimal_impulse,
                       torch.zeros_like(height_above_ground)) \
            + MSELOSS(optimal_impulse, F)


    # learning parameters
    models = [modelF, modelGH]
    losses = [lossF, lossGH]
    testlosses = [lossF] * len(models)
    catter = lambda x_, y_: torch.cat((x_, y_), dim=1)  # noqa
    mdata = [(lambda x_, y_: x_)] * len(models)
    ldata = [(lambda x_, y_: y_)] * (len(models) - 1) + [catter]
    mdata_t = [(lambda x_, y_: x_)] * len(models)
    ldata_t = [(lambda x_, y_: y_)] * len(models)
    # early_stop_epochs = [120, 50, 120, 120, 120, 120]

    legend = ['Baseline', 'ContactNets (Ours)']
    savef = ['modelF.pt', 'modelGH.pt']
    rates = [learning_rate] * len(models)
    opts = [0] * len(models)


    # early stopping parameters
    early_stop_epochs = [estop] * len(models)
    best_loss = [10000.] * len(models)
    best_epoch = [0] * len(models)
    end = [False, False, False]


    '''
    DATA SYNTHESIS
    '''
    # set range of initial velocities
    # center initial velocity v0 around impact
    v_center = g / 2 * dt - q0 / dt
    SC = .5
    v0min = v_center * (1 - SC)
    v0max = v_center * (1 + SC)

    STD = 0.1  # noise standard deviation

    if regenerate_data:
        # generate training data
        v0 = (v0max - v0min) * torch.rand(N, 1) + v0min
        xf = nominalModel(v0, GH)

        # generate test data
        v0_t = (v0max - v0min) * torch.rand(N, 1) + v0min
        xf_t = nominalModel(v0_t, GH)

        # generate plotting data
        v0_plot = torch.linspace(v0.min(), v0.max(), N * 100).unsqueeze(1)
        xf_plot = nominalModel(v0_plot, GH)

        # corrupt training and test data with gaussian noise
        v0 = v0 + (STD) * torch.randn(v0.shape)
        xf = xf + (STD) * torch.randn(xf.shape)

        v0_t = v0_t + (STD) * torch.randn(v0_t.shape)
        xf_t = xf_t + (STD) * torch.randn(xf_t.shape)



        '''
        LEARNING
        '''


        x = v0
        y = xf
        x_t = v0_t
        y_t = xf_t

        def permuteTensors(a, b):
            perm = torch.randperm(a.nelement())
            return (a[perm, :], b[perm, :])

        for (i, m) in enumerate(models):
            torch.save(m.state_dict(), savef[i])
            opts[i] = torch.optim.Adam(m.parameters(), lr=rates[i])

        for t in range(N_epoch):
            trained = False
            # randomly permute data
            (x, y) = permuteTensors(x, y)
            for (i, m) in enumerate(models):
                m.train()
                if not end[i]:
                    trained = True
                    for j in range(N_batches):
                        # get batch
                        samp_j = torch.range(0, N_batch - 1).long() + j * N_batch
                        y_pred = m(mdata[i](x[samp_j, :], y[samp_j, :]))

                        # get loss
                        lm = losses[i]
                        loss = lm(y_pred, ldata[i](x[samp_j, :], y[samp_j, :]))

                        # gradient step
                        opts[i].zero_grad()
                        loss.backward()
                        opts[i].step()
            if not trained:
                break
            if t % 2 == 1:
                print(t, best_loss)
                for (i, m) in enumerate(models):
                    if not end[i]:
                        # update test loss
                        m.eval()
                        y_pred_t = m(mdata_t[i](x_t, y_t))
                        lm = testlosses[i]
                        loss = lm(y_pred_t, ldata_t[i](x_t, y_t))

                        # save the model if it's the best_epoch so far
                        if best_loss[i] - loss.item() > 0.0:
                            best_loss[i] = loss.item()
                            best_epoch[i] = t
                            torch.save(m.state_dict(), savef[i])

                        # terminate tranining if no improvement in (early_stop_epochs) epochs
                        end[i] = end[i] or t - best_epoch[i] > early_stop_epochs[i]

        # save training data
        CVT = torch.cat((v0, xf), dim=1)
        np.savetxt('adam_comp_data.csv', CVT.detach().numpy(), delimiter=',')
    else:
        CVT = torch.tensor(np.loadtxt('adam_comp_data.csv', delimiter=',')).float()
        v0 = CVT[:, 0:1]
        xf = CVT[:, 1:3]
        CV = torch.tensor(np.loadtxt('adam_comp_mods.csv', delimiter=',')).float()
        v0_plot = CV[:, 0:1]
        xf_plot = CV[:, 1:3]

    # reload best models
    for (i, m) in enumerate(models):
        m.load_state_dict(torch.load(savef[i]), strict=False)

    # save models for plotting
    CV = torch.cat((v0_plot, xf_plot), dim=1)
    for m in models:
        m.eval()
        xfp = m(v0_plot)
        CV = torch.cat((CV, xfp), dim=1)
    np.savetxt('adam_comp_mods.csv', CV.detach().numpy(), delimiter=',')


    '''
    PLOTTING
    '''

    # matplotlib settings
    # fm = matplotlib.font_manager.json_load(
        # os.path.expanduser("~/.matplotlib/fontlist-v310.json"))
    # fm.findfont("serif", rebuild_if_missing=True)

    # rc('font', **{'family': ['Computer Modern Roman'], 'size': 10})
    # rc('figure', titlesize=14)
    # rcParams['mathtext.fontset'] = 'cm'
    # rcParams['mathtext.default'] = 'regular'
    # rc('text', usetex=True)
    # rc('legend', fontsize=10)
    # rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
    # plt.rc('axes', titlesize=14)     # fontsize of the axes title
    # plt.rc('axes', labelsize=14)    # fontsize of the x and y labels



    # styling
    LINEWIDTH = 2
    PENNRED = '#95001a'
    PENNBLUE = '#01256e'
    PENNDGRAY2 = '#44464b'  # noqa
    PENNLGRAY2 = '#e0e1e3'
    PENNLGRAY4 = '#bbbdc0'  # noqa
    PENNYELLOW = '#f2c100'

    colors = [PENNBLUE, PENNRED]

    def styleplot(fig, savefile, width=4, height=4):
        fig.set_size_inches(width, height)
        plt.tight_layout()
        # plt.gcf().subplots_adjust(bottom=0.15)
        fig.savefig(savefile, dpi=400)

    qf = xf[:, 0:1]
    vf = xf[:, 1:2]
    qfD = xf_plot[:, 0:1]
    vfD = xf_plot[:, 1:2]

    # construct legend
    lfinal = ['True System']
    for (i, m) in enumerate(models):
        lfinal = lfinal + legend[i:i + 1]
    lfinal = lfinal + ['Data']

    # plot position predicton
    fig = plt.figure(1)
    fig.suptitle("1D System Predictions")
    ax1 = plt.subplot(121)
    YMIN = -0.3
    ax1.fill_between(v0_plot.squeeze(), 0 * v0_plot.squeeze() + YMIN,
                     color=PENNLGRAY2, label='_nolegend_')
    plt.plot(v0_plot.numpy(), qfD.numpy(), linewidth=LINEWIDTH, color=PENNYELLOW)

    for (i, m) in enumerate(models):
        m.eval()
        xf_pred = m(v0_plot)
        qf_pred = xf_pred[:, 0:1]
        plt.plot(v0_plot.numpy(), qf_pred.detach().numpy(),
                 linewidth=LINEWIDTH, color=colors[i], linestyle='dashed')

    plt.scatter(v0, qf, color=PENNYELLOW)
    plt.legend(lfinal)

    plt.ylabel(r"Next Position $z'$")
    plt.xlabel(r"Initial Velocity $\dot z$")
    plt.title(r" ")
    ax1.text(4.05, -0.07, r'(below ground)', fontsize=12)
    plt.gca().set_xlim(torch.min(v0_plot), torch.max(v0_plot))
    plt.gca().set_ylim(YMIN, torch.max(qf) + 0.1)

    # plot velocity prediction
    ax1 = plt.subplot(122)
    plt.plot(v0_plot.numpy(), vfD.numpy(), linewidth=LINEWIDTH, color=PENNYELLOW)

    for (i, m) in enumerate(models):
        m.eval()
        xf_pred = m(v0_plot)
        vf_pred = xf_pred[:, 1:2]
        plt.plot(v0_plot.numpy(), vf_pred.detach().numpy(),
                 linewidth=LINEWIDTH, color=colors[i], linestyle='dashed')

    plt.scatter(v0, vf, color=PENNYELLOW)
    plt.ylabel(r"Next Velocity $\dot z'$")
    plt.xlabel(r"Initial Velocity $\dot z$")
    plt.title(r" ")
    plt.gca().set_xlim(torch.min(v0_plot), torch.max(v0_plot))
    # styleplot(fig,'PM_velocity.png')

    styleplot(fig, 'PM_config.png', width=8, height=5)


    # plot loss and loss gradient
    # construct ground heights
    NG = 1000
    GH_SCALE = dt * 10. * STD * SC
    ghs = torch.linspace(-GH_SCALE, GH_SCALE, NG)
    l1 = torch.zeros_like(ghs)
    l2 = torch.zeros_like(ghs)

    for i in range(NG):

        # get L2 loss
        mod_gh = GroundHeightQP(ghs[i].clone())
        gh = mod_gh.gh
        mod_gh.eval()
        xf_pred = mod_gh(v0)
        l1[i] = lossF(xf, xf_pred).clone().detach()
        l2[i] = lossGH(gh, torch.cat((v0, xf), dim=1)).clone().detach()

    lossdata = torch.cat((ghs.unsqueeze(1), l1.unsqueeze(1), l2.unsqueeze(1)), dim=1)
    np.savetxt('adam_comp_losses.csv', lossdata.detach().numpy(), delimiter=',')

    # normalize plots
    # gl1 /= gl1.max()
    # gl2 /= gl2.max()
    # l1 /= l1.max()
    # l2 /= l2.max()

    fig = plt.figure(3)
    plt.plot(ghs.numpy(), l1.detach().numpy(), linewidth=LINEWIDTH, color=PENNBLUE)
    plt.plot(ghs.numpy(), l2.detach().numpy(), linewidth=LINEWIDTH, color=PENNRED)
    plt.legend(['L2 Prediction', 'Mechanics-Based (Ours)'])
    plt.title('1D System Loss')
    plt.xlabel(r"Modeled Ground Height $\hat z_g$")
    styleplot(fig, 'PM_loss.png', height=5)


@click.command()
@click.option('--regenerate_data/--plot_only', default=True)
def main(regenerate_data: bool):
    generate_figure(regenerate_data)


if __name__ == "__main__": main()
