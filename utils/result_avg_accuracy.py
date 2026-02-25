
import json

test_dir = './results/bacteria_fishersam_gelu_pca_split_'
num_splits = 50

accuracy_sum = 0
patient_wise_accuracy_sum = 0

for x in range(num_splits):
	f = open(test_dir + str(x) + '/inference/test_results.json', 'r')
	result_json = json.load(f)
	accuracy_sum += result_json['test_acc']
	if result_json['test_acc'] > 0.5:
		patient_wise_accuracy_sum += 1
	f.close()

print('Single spectra accuracy: ' + str(accuracy_sum/num_splits))
#print('Patient-wise accuracy: ' + str(patient_wise_accuracy_sum/num_splits))