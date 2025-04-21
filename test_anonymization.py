"""
Test script for the Top Down Greedy Anonymization algorithm
"""
import sys
import os
import time
from functools import cmp_to_key

# Create minimal versions of required dependencies
class TreeNode:
    """
    Definition of a node in the generalization hierarchy
    """
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = []
        if parent:
            self.parent = [parent]

    def __repr__(self):
        return f"TreeNode({self.value})"

class NumRange:
    """
    Class for numerical attributes
    """
    def __init__(self, sort_value, value):
        self.sort_value = sort_value
        self.value = value

def cmp_str(s1, s2):
    """
    Compare two strings by their numerical value
    """
    return float(s1) - float(s2)

def get_num_list_from_str(value):
    """
    Convert string representation to list of numbers
    """
    if not value:
        return []
    try:
        if ',' in value:
            return value.split(',')
        else:
            return [value]
    except:
        return []

def create_demo_dataset():
    """
    Create a simple demonstration dataset
    """
    # Simple dataset with 3 attributes: age, education, and zipcode
    data = [
        ['30', 'Bachelor', '47906'],
        ['25', 'Master', '47307'],
        ['42', 'PhD', '47905'],
        ['35', 'Bachelor', '47905'],
        ['28', 'Master', '47307'],
        ['55', 'PhD', '47906'],
        ['38', 'Bachelor', '47302'],
        ['45', 'Master', '47905'],
        ['32', 'Bachelor', '47307'],
        ['50', 'PhD', '47302'],
        ['27', 'Bachelor', '47906'],
        ['22', 'Master', '47302'],
    ]
    return data

def create_attribute_trees():
    """
    Create generalization hierarchies for the attributes
    """
    # Numerical attribute: age (0-100)
    age_values = [str(i) for i in range(100)]
    age_range = NumRange(age_values, "0,99")

    # Categorical attribute: education
    education_tree = {}
    # Create hierarchy: specific degree -> degree level -> *
    # Bachelor, Master, PhD -> Degree -> *
    star_node = TreeNode("*")
    degree_node = TreeNode("Degree", star_node)
    bachelor_node = TreeNode("Bachelor", degree_node)
    master_node = TreeNode("Master", degree_node)
    phd_node = TreeNode("PhD", degree_node)
    
    education_tree["*"] = star_node
    education_tree["Degree"] = degree_node
    education_tree["Bachelor"] = bachelor_node
    education_tree["Master"] = master_node
    education_tree["PhD"] = phd_node

    # Categorical attribute: zipcode
    zipcode_tree = {}
    # Create hierarchy: full zipcode -> first 3 digits -> *
    star_node = TreeNode("*")
    
    # Group by first 3 digits
    zip_groups = {
        "473": TreeNode("473**", star_node),
    }
    
    # Add full zipcodes
    for zipcode in ["47302", "47307", "47905", "47906"]:
        prefix = zipcode[:3]
        if prefix in zip_groups:
            parent = zip_groups[prefix]
            zipcode_tree[zipcode] = TreeNode(zipcode, parent)
    
    # Add the group nodes and root
    zipcode_tree["*"] = star_node
    for prefix, node in zip_groups.items():
        zipcode_tree[f"{prefix}**"] = node
    
    return [age_range, education_tree, zipcode_tree]

# Import our anonymization algorithm
from update_anonymization import Top_Down_Greedy_Anonymization

def main():
    """
    Main function to test anonymization
    """
    # Create a sample dataset and attribute hierarchies
    data = create_demo_dataset()
    att_trees = create_attribute_trees()
    
    # Set k-anonymity parameter (k)
    k = 3
    
    print("Original dataset:")
    for row in data:
        print(row)
    
    # Run the anonymization algorithm
    print("\nRunning Top Down Greedy Anonymization with k =", k)
    start_time = time.time()
    
    # Call the anonymization algorithm
    result, metrics = Top_Down_Greedy_Anonymization(att_trees, data, k)
    
    end_time = time.time()
    
    # Output the results
    print("\nAnonymized dataset:")
    for row in result:
        print(row)
    
    print("\nMetrics:")
    ncp, runtime = metrics
    print(f"NCP: {ncp:.2f}%")
    print(f"Runtime: {runtime:.4f} seconds")
    print(f"Total time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main() 