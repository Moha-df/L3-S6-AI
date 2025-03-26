#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

DEBUG = True

###############################################################################

def entropy(df, target_name):
  """
  Evaluate the entropy of dataset df w.r.t. possible classes in list labels
  """
  # [which classes in labels] , [how many instances for each class]
  classes, nb_occ = np.unique(df[target_name], return_counts=True)
  df_len = np.sum(nb_occ)
  
  entropy = 0
  for k in range(len(classes)):
    nk = nb_occ[k]
    entropy += nk/df_len * np.log2(nk/df_len)
  
  return -entropy

def attribute_gain(df, attribute, target):
  """
  Compute the gain of attribute attribute for a binary split
    1. Data are sorted based on attribute value
    2. Split value: value of attribute at which the target
        changes for the first time
    3. Two partitions are created upon target change:
        instances for which value in [a < split_values] and in [a >= split_value]
    4. Gain is computed for this partitionning
  
  Parameters:
    df: dataframe to work on
    attribute: attribute whose gain is to be evaluated
    target: attribute (class to predict) for which we want to evaluate
            the significance of attribute attribute 
  Returns:
    gain, split_value, partitions
  """
  H_total    = entropy(df, target)
  df_sorted  = df.sort_values(by=attribute)
  df_sorted  = df_sorted.reset_index(drop=True)
  attr_value = df_sorted[attribute]
   
  t1 = df_sorted[target][0]
  partitions = df
  gain = 0
  split_value = None
   
  for i in range(1, df_sorted.shape[0]):
      if df_sorted[target][i] !=  t1:
          split_value  = df_sorted[attribute][i]
          partitions   = [df_sorted[attr_value < split_value],
                          df_sorted[attr_value >= split_value]]
          E_partitions = [entropy(partitions[j], target)
                          for j in range(len(partitions))]
          accu = 0.0
          for j in range(len(partitions)):
              accu += partitions[j].shape[0] / df_sorted.shape[0] * E_partitions[j]
          gain = H_total - accu
          break # exit loop once a target change has been observed
  return gain, split_value, partitions

def best_attribute(df, attributes, target):
  """
  Determines which attribute is the most significant to classify the data
  
  Parameters:
    df: data to work on
    attributes : attributes of the dataset (minus the target)
    target: attribut we want to predict
  Returns:
    best_attribute, best_gain, best_partitions, best_split_value
  """
  best_gain = 0.0
  best_partitions  = []
  best_split_value = 0.0
  best_attr = None
  
  for a in attributes:
    gain, split_value, partitions = attribute_gain(df, a, target)
    
    if (gain > best_gain):
        best_gain, best_split_value, best_partitions, best_attr = \
                        gain, split_value, partitions, a
  
  return best_attr, best_gain, best_partitions, best_split_value

###############################################################################

class DecisionTree:
  def __init__(self, attribute=None, split_value=None,
               left_branch=None, right_branch=None,
               prediction=None, isLeaf=False): 
    self.attribute    = attribute
    self.split_value  = split_value
    self.isLeaf = isLeaf
    self.left_branch  = left_branch
    self.right_branch = right_branch
    self.prediction   = prediction
      
  ## Stuff to pretty-print the tree / nodes
  def node_result(self, spacing=''):
    s = ''
    for v in range(len(self.prediction.values)):
        s += ' Class ' + str(self.prediction.index[v]) + ' Count: ' +\
          str(self.prediction.values[v]) + '\n' + spacing
    return s
  
  def __str__(self):
    return '<' + str(self.split_value) + '\n' + self.node_result() + '>'
  
  def __repr__(self):
    return str(self.split_value) + ' ' + self.node_result()
  
  def print_tree(self, node=None, spacing='', depth=0):
    if node == None: node = self
    
    if node is None:
        return
    if node.isLeaf:
        print(spacing + node.node_result(spacing))
        return
    print(f'{spacing}[Attribute: {node.attribute}  Split value: {node.split_value}]')
    
    print(spacing + '> True')
    self.print_tree(node.left_branch, spacing + '-', depth+1)
    
    print(spacing + '> False')
    self.print_tree(node.right_branch, spacing + '-', depth+1)
    
    return None
  
  ## Learn the decision tree on data
  def fit(self, df, target, attributes, depth=0, max_depth=10):
    """
    Learn the tree using the data df
    Parameters:
      data: data to learn from (pandas dataframe)
      target: attribute to predict (string)
      attributes: attributes to use to build the tree (list)
    Returns:
      Root of the tree (DecisionTree object)
    """
    #print(f"DEBUG: Entering fit function, depth={depth}")
    
    # Base case 1: Maximum depth reached
    if depth >= max_depth:
        #print(f"DEBUG: Max depth reached at depth={depth}")
        node = DecisionTree(isLeaf=True)
        node.prediction = df[target].value_counts()
        return node
    
    # Base case 2: All instances belong to the same class
    if len(df[target].unique()) == 1:
        #print(f"DEBUG: All instances same class: {df[target].unique()[0]}")
        node = DecisionTree(isLeaf=True)
        node.prediction = df[target].value_counts()
        return node
    
    # Base case 3: No more attributes to split on
    if len(attributes) == 0:
        #print("DEBUG: No more attributes to split on")
        node = DecisionTree(isLeaf=True)
        node.prediction = df[target].value_counts()
        return node
    
    # Find the best attribute to split on
    #print(f"DEBUG: Finding best attribute among {attributes}")
    best_attr, best_gain, best_partitions, best_split_value = best_attribute(df, attributes, target)
    #print(f"DEBUG: Best attribute is {best_attr} with gain {best_gain}, split value {best_split_value}")
    
    # If no good split was found (gain = 0), create a leaf node
    if best_gain == 0:
        #print("DEBUG: No good split found (gain=0)")
        node = DecisionTree(isLeaf=True)
        node.prediction = df[target].value_counts()
        return node
    
    # Create a new decision node
    #print(f"DEBUG: Creating decision node for attribute {best_attr}")
    node = DecisionTree(attribute=best_attr, split_value=best_split_value)
    
    # Build the left subtree (for instances where attribute < split_value)
    left_partition = best_partitions[0]
    #print(f"DEBUG: Left partition size: {len(left_partition)}")
    if len(left_partition) > 0:
        #print(f"DEBUG: Building left subtree for {best_attr} < {best_split_value}")
        node.left_branch = self.fit(left_partition, target, attributes, depth+1, max_depth)
    else:
        # If the partition is empty, create a leaf with the majority class
        #print("DEBUG: Left partition empty, creating leaf")
        node.left_branch = DecisionTree(isLeaf=True)
        node.left_branch.prediction = df[target].value_counts()
    
    # Build the right subtree (for instances where attribute >= split_value)
    right_partition = best_partitions[1]
    #print(f"DEBUG: Right partition size: {len(right_partition)}")
    if len(right_partition) > 0:
        #print(f"DEBUG: Building right subtree for {best_attr} >= {best_split_value}")
        node.right_branch = self.fit(right_partition, target, attributes, depth+1, max_depth)
    else:
        # If the partition is empty, create a leaf with the majority class
        #print("DEBUG: Right partition empty, creating leaf")
        node.right_branch = DecisionTree(isLeaf=True)
        node.right_branch.prediction = df[target].value_counts()
    
    # Set prediction for node (used for printing)
    node.prediction = df[target].value_counts()
    
    # Return the node we've created
    #print(f"DEBUG: Returning node at depth={depth}")
    return node

  def predict(self, instance, tree=None, parent_attribute=None, spacing=""):
    """
    Print the class of instance instance
      Explores the tree recursively until a leaf is reached
      If leaf contains several class, prediction is the most represented class
    
    Returns: leaf node prediction, predicted class, class proportion in leaf node
    """
    #print(f"DEBUG: Entering predict function")
    
    # Initialize tree to self if not provided (first call)
    if tree is None:
        tree = self
        #print(f"DEBUG: Starting at root node")
    
    # Base case: we've reached a leaf node
    if tree.isLeaf:
        #print(f"DEBUG: Reached leaf node")
        print(f"{spacing}-> Node prediction:\n {tree.prediction}")
        
        # Get the class with the highest count
        predicted_class = tree.prediction.idxmax()
        
        # Calculate the proportion of instances belonging to the predicted class
        total_instances = tree.prediction.sum()
        proportion = (tree.prediction[predicted_class] / total_instances) * 100
        
        #print(f"DEBUG: Predicting class {predicted_class} with proportion {proportion}%")
        return tree.prediction, predicted_class, proportion
    
    # Print the current node's attribute and split value
    print(f"{spacing}Node attribute: {tree.attribute}  Split value: {tree.split_value}")
    
    # Get the instance's value for the current attribute
    attribute_value = instance[tree.attribute].values[0]  # Extract the value from the pandas Series
    print(f"{spacing}Instance's value for {tree.attribute}: {attribute_value}")
    
    # Recursive case: traverse left or right depending on the attribute value
    if attribute_value < tree.split_value:
        #print(f"DEBUG: Going left ({attribute_value} < {tree.split_value})")
        return self.predict(instance, tree.left_branch, tree.attribute, spacing + "-")
    else:
        #print(f"DEBUG: Going right ({attribute_value} >= {tree.split_value})")
        return self.predict(instance, tree.right_branch, tree.attribute, spacing + "-")
###############################################################################



###############################################################################
def test_DecisionTree(tree, test_data):
    """
    Tests the decision tree tree on some test data
    Parameters :
        tree: trained decision tree (DecisionTree)
        test_data: set of instances from the dataset (dataframe)
    """
    if tree == None: return None
    
    tree.print_tree()

    # Test the tree on a bunch of random instances
    for i in range(4):
        instance = test_data.sample()
        print(f"** Instance to predict: {instance[target]}\n")
        _, predicted_class, proportion = tree.predict(instance)
        print(f"Predicted class: {predicted_class} ({proportion})\n")
    return None



def main():
  #print(f"DEBUG: Entering main function")
  # Load the data
  df = pd.read_csv('iris.csv')
  
  # Set a portion of data aside for testing
  df_test  = df.sample(frac=0.15, random_state=42) # 22 instances
  df_train = df.drop(df_test.index) # 128 instances
  
  # Get attributes and target
  target = 'class'
  attributes = list(df_train.columns)[:-1] # attributes minus the target

  # Instanciate and train a decision tree on df
  tree = DecisionTree()
  tree = tree.fit(df_train, target, attributes)
  
  # Now, let's test that tree
  test_DecisionTree(tree, df_test)

  #print(f"DEBUG: returnning from mains")
  
  return None

if __name__ == "__main__":
  main()