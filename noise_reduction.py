import librosa
import librosa.display
import soundfile as sf
import os
from maad import sound, util
from utils import * 
import yaml
from vmdpy import VMD
import numpy as np

def towsey(signal, sr, mode, n_mels=128, fmax=8000):
	""" 
	Remove background noise from signal using technique proposed by Towsey in 2013.

	Parameters 
	---------- 
	signal : 1D numpy array  
		Original signal 

	mode : str, optional, default is 'ale'
		Defines the mode used to remove the noise using scikit-maad
		Possible values for mode are :
		- 'ale' : Adaptative Level Equalization algorithm [Lamel & al. 1981]
		- 'median' : subtract the median value
		- 'mean' : subtract the mean value (DC)

	n_mels : number of Mel bands to generate, used in librosa mel_spectrogram function

	fmax : max frequency with information, used in librosa mel_spectrogram function   

	Returns 
	------- 
	treated_signal : 1d numpy array 
		Signal after denoising 

	References
	----------
	.. [1] Towsey, M., 2013. Noise Removal from Wave-forms and Spectrograms Derived from Natural Recordings of the Environment. Queensland University of Technology, Brisbane
	.. [2] Scikit-maad: https://github.com/scikit-maad/scikit-maad/blob/production/maad/sound/spectral_subtraction.py
	.. [3] Librosa: https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
					https://librosa.org/doc/main/generated/librosa.power_to_db.html
	"""

	S_dB = signal_to_spec(signal, sr)

	S_dB_Towsey, _ = sound.remove_background_along_axis(S_dB, mode=mode)

	return S_dB_Towsey

def spectral_subtraction(signal, sr):

	S = signal_to_spec(signal, sr)
	S_dB = util.power2dB(S)

	S_dB_spec_subt, noise_profile, _ = sound.remove_background(S_dB)

	return S_dB_spec_subt

def pcen(signal, sr):
	# print('entrou pcen')
	# print(signal)
	S = signal_to_spec(signal, sr)
	# print(S)
	# print("has NAN original SPEC: ", np.isnan(S).any())
	# S_dB = util.power2dB(S) + 96
	# print(S_dB)
	# S_dB[S_dB == float("-inf")] = 0
	# print("has NAN original SPEC DB: ", np.isnan(S_dB).any())
	# print(S_dB)
	# breakpoint()
	S_dB_Pcen, _, _ = sound.pcen(S, gain = 0.2, bias = 0.5, power = 0.1)#bias = 100, power = 1)
	# print("has NAN original SPEC PCEN: ", np.isnan(S_dB_Pcen).any())
	# print('PCEN\n', S_dB_Pcen)
	return S_dB_Pcen
	

def vmd(sig, alpha, tau, K, DC, init, tol, n_mels=128, fmax=8000, mode='ale'):

	u, u_hat, omega = VMD(sig, alpha, tau, K, DC, init, tol)

	min_delta = float("inf")
	pos = -1
	total_sig = [0] * len(u[0])
	for i, signal in enumerate(u):
		delta = max(signal) - min(signal)
		if(delta < min_delta):
			min_delta = delta
			pos = i
		total_sig += signal
	new_signal = total_sig - u[pos]

	# Visualize decomposed modes
	# print("-----------------")
	# # print(new_sig)
	# print(total_sig)
	# plt.figure()
	# plt.subplot(3,1,1)
	# plt.plot(sig)
	# plt.title('Original signal')
	# plt.xlabel('time (s)')
	# plt.subplot(3,1,2)
	# plt.plot(u.T)
	# plt.title('Decomposed modes')
	# plt.xlabel('time (s)')
	# plt.legend(['Mode %d'%m_i for m_i in range(u.shape[0])])
	# plt.subplot(3,1,3)
	# plt.plot(total_sig)
	# plt.title('New Signal')
	# plt.xlabel('time (s)')
	# plt.tight_layout()
	# plt.show()

	return new_signal

##### Implementation of NMF function and divergence by ZahraBenslimane available at https://gist.github.com/ZahraBenslimane ######
def divergence(V,W,H, beta = 2):
	
	"""
	beta = 2 : Euclidean cost function
	beta = 1 : Kullback-Leibler cost function
	beta = 0 : Itakura-Saito cost function
	""" 
	
	if beta == 0 : return np.sum( V/(W@H) - math.log10(V/(W@H)) -1 )
	
	if beta == 1 : return np.sum( V*math.log10(V/(W@H)) + (W@H - V))
	
	if beta == 2 : return 1/2*np.linalg.norm(W@H-V)

def NMF(V, S, beta = 2,  threshold = 0.05, MAXITER = 5000, display = True , displayEveryNiter = None): 
	
	"""
	inputs : 
	--------
	
		V		 : Mixture signal : |TFST|
		S		 : The number of sources to extract
		beta	  : Beta divergence considered, default=2 (Euclidean)
		threshold : Stop criterion 
		MAXITER   : The number of maximum iterations, default=1000
		display   : Display plots during optimization : 
		displayEveryNiter : only display last iteration 
															
	
	outputs :
	---------
	  
		W : dictionary matrix [KxS], W>=0
		H : activation matrix [SxN], H>=0
		cost_function : the optimised cost function over iterations
	   
   Algorithm : 
   -----------
   
	1) Randomly initialize W and H matrices
	2) Multiplicative update of W and H 
	3) Repeat step (2) until convergence or after MAXITER 
	
	   
	"""
	counter  = 0
	cost_function = []
	beta_divergence = 1
	
	K, N = np.shape(V)
	
	# Initialisation of W and H matrices : The initialization is generally random
	W = np.abs(np.random.normal(loc=0, scale = 2.5, size=(K,S)))	
	H = np.abs(np.random.normal(loc=0, scale = 2.5, size=(S,N)))
	
	# Plotting the first initialization
	if display == True : plot_NMF_iter(W,H,beta,counter)


	while beta_divergence >= threshold and counter <= MAXITER:
		
		# Update of W and H
		H *= (W.T@(((W@H)**(beta-2))*V))/(W.T@((W@H)**(beta-1)) + 10e-10)
		W *= (((W@H)**(beta-2)*V)@H.T)/((W@H)**(beta-1)@H.T + 10e-10)
		
		
		# Compute cost function
		beta_divergence =  divergence(V,W,H, beta = 2)
		cost_function.append( beta_divergence )
		
		if  display == True  and counter%displayEveryNiter == 0  : plot_NMF_iter(W,H,beta,counter)

		counter +=1
	
	# if counter -1 == MAXITER : print(f"Stop after {MAXITER} iterations.")
	# else : print(f"Convergeance after {counter-1} iterations.")
		
	return W,H, cost_function

def NMF_tech(signal, sr, S = 2):

	# S_dB = signal_to_spec(signal, sr)

	# K, N = np.shape(S_dB)

	sound_stft = librosa.stft(signal, n_fft = 512, hop_length = 256)
	S_dB = np.abs(sound_stft)


	beta = 1

	W, H, cost_function = NMF(S_dB,S,beta = beta, threshold = 0.05, MAXITER = 500, display = False, displayEveryNiter = 1000)

	target_spec = []	

	min_median_value = float("inf")

	for i in range(S):
		WsHs = W[:,[i]]@H[[i],:]
		filtered_spectrogram = W[:,[i]]@H[[i],:] /(W@H) * S_dB

		D = librosa.amplitude_to_db(filtered_spectrogram, ref = np.max)
		median_val = np.average(D)
		if(median_val < min_median_value):
			# print("entrou")
			min_median_value = median_val
			target_spec = D
	# return filtered_spectrogram
	return target_spec

def apply_technique_NR(signal, sr, technique):
	with open('new_config.yaml') as f:
		cfg = yaml.load(f, Loader=yaml.FullLoader)
	if(technique == 'nmf'):
		return NMF_tech(signal, sr)
	elif(technique == 'vmd'):
		vmd_info = cfg['techniques']['vmd']['params']
		alpha, tau, K, DC, init, tol = vmd_info['alpha'], vmd_info['tau'], vmd_info['K'], vmd_info['DC'], vmd_info['init'], vmd_info['tol']
		return vmd(signal, alpha, tau, K, DC, init, tol)
	
	elif(technique == 'towsey'):
		towsey_info = cfg['techniques']['towsey']['params']
		mode = towsey_info['mode']
		# print("passou aqui")
		return towsey(signal, sr, mode)
		# print("aux: ", aux)
		# return aux
	
	elif(technique == 'pcen'):
		return pcen(signal, sr)
	elif(technique == 'spectral_subtraction'):
		return spectral_subtraction(signal, sr)
	elif(technique == 'none'):
		return signal
		
