import numpy as np
import matplotlib.pyplot as plt; import matplotlib
matplotlib.use('Qt5Agg')
import os

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    folder_path = current_dir+'/2_Output/tac_vf_label/'
    filenames = os.listdir(folder_path)

    IDs = []
    for name in filenames:
        if name.startswith('tac'):
            id = float(name.split('label')[-1].split('.npz')[0])
            IDs.append(id)

    Results = []
    for id in IDs:
        data = np.load(folder_path+'tac_nadir_vf_label'+str(id)+'.npz')['array1']
        names = np.load(folder_path+'tac_nadir_vf_label'+str(id)+'.npz')['array2']
        vf = data[-2,:]+data[-1,:]
        # plt.figure(); plt.hist(vf,bins=20)
        fractions = [0,0.2,0.4,0.6,0.8]

        tac_3zone = []
        for frc in fractions:
            mask = (vf>=frc) & (vf<frc+0.2)
            tac = data[:,mask]
            tac[0,:][tac[0,:]<0]=np.nan
            tac_uc = np.nanmean(tac[0,~np.isnan(tac[1,:])])
            tac_ue = np.nanmean(tac[0,~np.isnan(tac[2, :])])
            tac_ra = np.nanmean(tac[0,~np.isnan(tac[3, :])])
            tac_3zone.append([tac_uc,tac_ue,tac_ra])

        tac_3zone_arr = np.array(tac_3zone)
        Results.append(tac_3zone_arr)
        print(id)
    Results_arr = np.array(Results)

    # Modify bar plot to add dots pattern
    fig, axs = plt.subplots(5, 3, figsize=(7.5* 0.9, 1.8* 0.9*5),width_ratios=[2,1,1])
    for i in range(5):
        bars = axs[i,0].bar(np.linspace(0, 2, 3), np.nanmean(Results_arr[:,i,:],axis=0),
                      yerr=np.nanstd(Results_arr[:,i,:], axis=0) * 0.25,
                      width=0.30,
                      color=['#2166ac', '#67a9cf', '#b2182b'])
    
        axs[i,0].set_xticks([0, 1, 2], ['UC', 'UE', 'RA'])
        axs[i, 0].set_ylabel('TAC')

    
        mask_interface = ~np.isnan(Results_arr[:,i,0]) & ~np.isnan(Results_arr[:,i,1])
        mask_rural = ~np.isnan(Results_arr[:,i,0]) & ~np.isnan(Results_arr[:,i,2])
        x_interface = Results_arr[mask_interface,i, 0]
        y_interface = Results_arr[mask_interface,i, 1]
    
        # Separate points above and below 1:1 line
        above_mask_interface = y_interface > x_interface
        below_mask_interface = ~above_mask_interface
        print(above_mask_interface.sum()/above_mask_interface.shape[0], above_mask_interface.sum()/above_mask_interface.shape[0])

        # Plot points with different colors
        axs[i,1].plot(x_interface[above_mask_interface], y_interface[above_mask_interface],
                   'o', mfc='none', color='C0')
        axs[i,1].plot(x_interface[below_mask_interface], y_interface[below_mask_interface],
                   'o', mfc='none', color='C3')
    
        # Add 1:1 and fit lines
        min_val = min(x_interface.min(), y_interface.min())
        max_val = max(x_interface.max(), y_interface.max())
        axs[i,1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        z = np.polyfit(x_interface, y_interface, 1)
        p = np.poly1d(z)
        # axs[i,1].plot(x_interface, p(x_interface), 'k-', alpha=0.7)
    
        # For urban core vs rural background plot
        x_rural = Results_arr[mask_rural, i, 0]
        y_rural = Results_arr[mask_rural,i, 2]
    
        # Separate points above and below 1:1 line
        above_mask_rural = y_rural > x_rural
        below_mask_rural = ~above_mask_rural
        print(above_mask_rural.sum()/above_mask_rural.shape[0], below_mask_rural.sum()/above_mask_rural.shape[0])
    
        # Plot points with different colors
        axs[i,2].plot(x_rural[above_mask_rural], y_rural[above_mask_rural],
                   'o', mfc='none', color='C0')
    
        axs[i,2].plot(x_rural[below_mask_rural], y_rural[below_mask_rural],
                   'o', mfc='none', color='C3')
    
        # Add 1:1 and fit lines
        min_val = min(x_rural.min(), y_rural.min())
        max_val = max(x_rural.max(), y_rural.max())
        axs[i,2].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        z = np.polyfit(x_rural, y_rural, 1)
        p = np.poly1d(z)
        # axs[i,2].plot(x_rural, p(x_rural), 'k-', alpha=0.7)

        axs[i, 0].set_ylim([0.10,0.24])
        axs[i,1].set_xlabel('TAC$_{UC}$')
        axs[i,1].set_ylabel('TAC$_{UE}$')
        axs[i,2].set_xlabel('TAC$_{UC}$')
        axs[i,2].set_ylabel('TAC$_{RA}$')

    figToPath = current_dir + '/4_Figures/sensitivity_tac_3zones_bar'
    fig.tight_layout(h_pad=0.1)
    fig.savefig(figToPath, dpi=900)