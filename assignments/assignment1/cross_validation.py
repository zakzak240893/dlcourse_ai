import numpy as np

class CV_Split:
    
    
    
    def __init__(self, train_dataset, folds_number = 5):
        self.folds_number = folds_number
        self.dataset = train_dataset
        self.dataset_shape = self.dataset.shape
        
    
    def _get_elements_number(self):
        return self.dataset.shape[0] // self.folds_number * self.folds_number
        
    def _get_splitted_shape(self, shape):
        shape_reminder = list(shape[1:])
        fold_element_num = self._get_elements_number() / self.folds_number
        return tuple([self.folds_number, int(fold_element_num)] + shape_reminder)
    
    def _get_train_cv_shape(self, shape):
        shape_reminder = list(shape[2:])
        fold_element_num = self._get_elements_number() / self.folds_number
        return tuple([int((self.folds_number - 1) * fold_element_num)] + shape_reminder)

    def _get_test_cv_shape(self, shape):
        #print("shape:",shape)
        shape_reminder = list(shape[1:])
        #print("shape_reminder:",shape_reminder)
        fold_element_num = self._get_elements_number() / self.folds_number
        #print("output:", tuple([int(fold_element_num)] + shape_reminder))
        return tuple([int(fold_element_num)] + shape_reminder)
        #return tuple(shape_reminder)
        
    def get_fold_train_test(self):
        dataset_split_shape = self._get_splitted_shape(self.dataset_shape)
        elem_num = self._get_elements_number()
        dataset_split = self.dataset[:elem_num].reshape(dataset_split_shape)
        for i in range(self.folds_number):
            cv_train = np.delete(dataset_split, i, axis = 0).reshape(self._get_train_cv_shape(dataset_split.shape))
            #dataset_split[i].shape
            cv_test = dataset_split[i].reshape(self._get_test_cv_shape(dataset_split[i].shape))
            yield cv_train, cv_test