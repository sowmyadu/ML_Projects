import numpy as np
import utils as Util

"""
The idea is pretty straight forward. Just the same idea with you studied from lecture about how to build a decision tree. In each node, if it's a leaf (based on three stop conditions from lecture), you stop and return. If it's not a leaf node, first we need to decide which attribute we will use to split this node (so we need to calculate information gain one by one and chose the largest information gain). Once we decided how to split current tree node, we generate child node one by one according to the tie break. Then you go to child nodes and try to split child nodes recursively.
"""
class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True
        
        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        #split based on IG
        #get unique values of a feature
        #iterate feature wise
        
        #for parent entropy calculation
        #print(self.labels)
        #print(self.features)
        featuresT = np.transpose(self.features)
        inf_gain_list = []
        max_gain = 0
        for i in range(len(featuresT)):
            p_ent = 0
            values = np.unique(featuresT[i])
            print("Type of values is ", type(values))
            num_values = len(values)
            print("Num of unique values: ",num_values) #child nodes
            num_examples = len(featuresT[i])
            for label in np.unique(self.labels):
                p_ent += -np.float((self.labels.count(label)))/num_examples*np.log2(np.float((self.labels.count(label)))/num_examples)
            branches = {}
            print(values) #attribute values
            for j in range(len(featuresT[i])):
                value = featuresT[i][j]
                if branches.get(value) is None:
                    branches[value] = {}
                    branches[value][self.labels[j]] = 1
                    #print(branches[value][self.labels[j]])
                    #print(branches.get(value))
                elif branches.get(value).get(self.labels[j]) is None:
                    branches.get(value)[self.labels[j]] = 1
                    #print(branches.get(value)[self.labels[j]])
                else:
                    branches[value][self.labels[j]] = branches.get(value).get(self.labels[j]) + 1
            #print(branches)
            #branches.get(value).append(self.labels[j])
            """for j in range(len(featuresT[i])):
                value = featuresT[i][j]
                if branches.get(value) is None:
                    branches[value] = 1
                    print("created new key")
                elif branches.get(value)>0:
                    branches[value] = branches.get(value) +1
            print(branches)"""
            branchesList = []
            j = 0
            for key in branches:
                branchesList.append([])
                #print(key)
                for i in range(self.num_cls):
                    #print(branches[key][i])
                    if branches.get(key).get(i) is None:
                        branchesList[j].append(0)
                    elif branches[key][i] > 0:
                        branchesList[j].append(branches[key][i])
                j = j+1
            #print(branchesList)    
            inf_gain = Util.Information_Gain(p_ent,branchesList)
            if inf_gain > max_gain:
                max_gain = inf_gain
                selected_feature_index = i
            #inf_gain_list.append(inf_gain)
            #print(inf_gain," is the information gain")    
        #selected_feature_index = inf_gain_list.index(max(inf_gain_list))
        #print(selected_feature_index)
        #split into children nodes
        
        self.dim_split = selected_feature_index
        self.feature_uniq_split = np.sort(values)
        
        for attribute in values:
            attribute_label = []
            featureList = []
            for i in range(len(featuresT[selected_feature_index])):
                if featuresT[selected_feature_index][i] == attribute:
                    attribute_label.append(self.labels[i])
                    featureList.append(self.features[i])
            for i in featureList:
                del i[selected_feature_index]
            #print(featureList)
            num_cls = np.unique(attribute_label).size
            #print(num_cls, "::" , len(attribute_label))
            self.children.append(TreeNode(featureList, attribute_label,num_cls))
        for child in self.children:
            if child.splittable:
                child.split()
        #print(self.children)
        
        #raise NotImplementedError

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        
        #self.splittable = false :: is a leaf node
        #value of feature to traverse
        
        raise NotImplementedError
