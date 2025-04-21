""" 
Main module of top down greedy anonymization algorithm 
""" 
import random 
import time 
import operator 
from functools import cmp_to_key 

from models.numrange import NumRange 
from utils.utility import cmp_str, get_num_list_from_str 

__DEBUG = False 
QI_LEN = 5 
GL_K = 0 
RESULT = [] 
ATT_TREES = [] 
QI_RANGE = [] 
ROUNDS = 5  
IS_CAT = [] 

class Partition: 
    """ 
    Class for Group, which is used to keep records 
    Store tree node in instances. 
    self.member: records in group 
    self.middle: save the generalization result of this partition 
    """ 
 
    def __init__(self, data, middle): 
        """Initialize with data and middle""" 
        self.can_split = True 
        self.member = data[:] 
        self.middle = middle[:] 
 
    def __len__(self): 
        """Return the number of records in partition""" 
        return len(self.member) 
 
def NCP(record): 
    """Compute Certainty Penalty of records""" 
    record_ncp = 0.0 
    for i in range(QI_LEN): 
        if not IS_CAT[i]: 
            value_ncp = 0 
            try: 
                # If it's already a single numeric value
                float(record[i]) 
            except ValueError: 
                # If it's a range in format "min,max"
                split_number = record[i].split(',') 
                if len(split_number) >= 2:
                    value_ncp = float(split_number[1]) - float(split_number[0]) 
            # Normalize by the range
            if QI_RANGE[i] > 0:
                value_ncp = value_ncp * 1.0 / QI_RANGE[i] 
            record_ncp += value_ncp 
        else: 
            # For categorical, use the level in the hierarchy
            if record[i] in ATT_TREES[i]:
                record_ncp += len(ATT_TREES[i][record[i]]) * 1.0 / QI_RANGE[i] 
    return record_ncp 
 
def NCP_dis(record1, record2): 
    """ 
    use the NCP of generalization record1 and record2 as distance 
    """ 
    mid = middle_record(record1, record2) 
    return NCP(mid), mid 
 
def NCP_dis_merge(partition, addition_set): 
    """ 
    merge addition_set to current partition, 
    update current partition.middle 
    """ 
    mid = middle_group(addition_set) 
    mid = middle_record(mid, partition.middle) 
    return (len(addition_set) + len(partition)) * NCP(mid), mid 
 
def NCP_dis_group(record, partition): 
    """ 
    compute the NCP of record and partition 
    """ 
    mid = middle_record(record, partition.middle) 
    ncp = NCP(mid) 
    return (1 + len(partition)) * ncp 
 
def middle_record(record1, record2): 
    """ 
    get the generalization result of record1 and record2 
    """ 
    mid = [] 
    for i in range(QI_LEN): 
        if not IS_CAT[i]: 
            # Handle numeric attributes
            split_number = [] 
            split_number.extend(get_num_list_from_str(record1[i])) 
            split_number.extend(get_num_list_from_str(record2[i])) 
            if not split_number:
                # Handle empty lists
                mid.append('0')
                continue

            split_number.sort(key=cmp_to_key(cmp_str)) 
            # avoid 2,2 problem 
            if split_number[0] == split_number[-1]: 
                mid.append(str(split_number[0])) 
            else: 
                mid.append(str(split_number[0]) + ',' + str(split_number[-1])) 
        else: 
            # Handle categorical attributes
            mid.append(LCA(record1[i], record2[i], i)) 
    return mid 
 
def middle_group(group_set): 
    """ 
    get the generalization result of the group 
    """ 
    if not group_set:
        return []
        
    len_group_set = len(group_set) 
    mid = group_set[0] 
    for i in range(1, len_group_set): 
        mid = middle_record(mid, group_set[i]) 
    return mid 
 
def LCA(u, v, index): 
    """ 
    get lowest common ancestor of u, v on generalization hierarchy (index) 
    """ 
    gen_tree = ATT_TREES[index] 
    
    # Handle case where values are not in the tree
    if u not in gen_tree or v not in gen_tree:
        return '*'
        
    # don't forget to add themselves (other the level will be higher) 
    u_parent = list(gen_tree[u].parent) 
    u_parent.insert(0, gen_tree[u]) 
    v_parent = list(gen_tree[v].parent) 
    v_parent.insert(0, gen_tree[v]) 
    min_len = min(len(u_parent), len(v_parent)) 
    
    if min_len == 0: 
        return '*' 
        
    last = -1 
    for i in range(min_len): 
        pos = - 1 - i 
        if u_parent[pos] != v_parent[pos]: 
            break 
        last = pos 
        
    if last == -1:
        return '*'
        
    return u_parent[last].value 
 
def get_pair(partition): 
    """ 
    To get max distance pair in partition, we need O(n^2) running time. 
    The author proposed a heuristic method: random pick u and get max_dis(u, v) 
    with O(n) running time; then pick max(v, u2)...after run ROUNDS times. 
    the dis(u, v) is nearly max. 
    """ 
    len_partition = len(partition) 
    
    # Use exhaustive search for small partitions
    if len_partition <= 50:   
        max_dis = -1 
        best_pair = (0, 1) 
        
        # Check all possible pairs 
        for i in range(len_partition): 
            for j in range(i+1, len_partition): 
                rncp, _ = NCP_dis(partition.member[i], partition.member[j]) 
                if rncp > max_dis: 
                    max_dis = rncp 
                    best_pair = (i, j) 
        return best_pair 
     
    # Standard approach for larger partitions - random sampling
    u = random.randrange(len_partition)
    for i in range(ROUNDS): 
        max_dis = -1 
        max_index = 0 
        
        for j in range(len_partition): 
            if j != u: 
                rncp, _ = NCP_dis(partition.member[j], partition.member[u]) 
                if rncp > max_dis: 
                    max_dis = rncp 
                    max_index = j 
                    
        v = max_index
        u = v  # For next round
        
    return (u, v) 
 
def distribute_record(u, v, partition): 
    """ 
    Distribute records based on NCP distance. 
    Records will be assigned to nearer group. 
    """ 
    record_u = partition.member[u][:] 
    record_v = partition.member[v][:] 
    u_partition = [record_u] 
    v_partition = [record_v] 
    
    remain_records = [item for index, item in enumerate(partition.member)  
                     if index not in {u, v}] 
     
    # Sort records by NCP distance difference for better distribution
    record_dists = [] 
    for record in remain_records: 
        u_dis, _ = NCP_dis(record_u, record) 
        v_dis, _ = NCP_dis(record_v, record) 
        record_dists.append((record, u_dis - v_dis)) 
     
    # Sort by distance difference to get more balanced partitions 
    record_dists.sort(key=lambda x: x[1]) 
     
    # Distribute records 
    for record, dist_diff in record_dists: 
        if dist_diff > 0: 
            v_partition.append(record) 
        else: 
            u_partition.append(record) 
     
    return [Partition(u_partition, middle_group(u_partition)), 
            Partition(v_partition, middle_group(v_partition))] 
 
def balance(sub_partitions, index): 
    """ 
    Two kinds of balance methods. 
    1) Move some records from other groups 
    2) Merge with nearest group 
    """ 
    less = sub_partitions.pop(index) 
    more = sub_partitions.pop() 
    all_length = len(less) + len(more) 
    require = GL_K - len(less) 
     
    # First method - move records from one partition to another
    dist = {} 
    for i, record in enumerate(more.member): 
        dist[i], _ = NCP_dis(less.middle, record) 
 
    sorted_dist = sorted(dist.items(), key=operator.itemgetter(1)) 
     
    # Try to find optimal number of records to move 
    best_ncp = float('inf') 
    best_config = None 
     
    # Try different numbers of records (not just minimum required) 
    for adj in range(5):  # Try moving 0-4 additional records 
        # Don't take too many records 
        extra = require + adj 
        if extra >= len(more.member) - GL_K + 1: 
            continue 
             
        nearest_index = [t[0] for t in sorted_dist[:extra]] 
        addition_set = [more.member[i] for i in nearest_index] 
        remain_set = [more.member[i] for i in range(len(more.member)) 
                      if i not in nearest_index] 
         
        # Skip if resulting partitions would be invalid 
        if len(remain_set) < GL_K: 
            continue 
             
        # Calculate NCP 
        temp_less = Partition(less.member + addition_set, []) 
        temp_less.middle = middle_group(temp_less.member) 
        temp_more = Partition(remain_set, []) 
        temp_more.middle = middle_group(remain_set) 
         
        total_ncp = len(temp_less) * NCP(temp_less.middle) + len(temp_more) * NCP(temp_more.middle) 
         
        if total_ncp < best_ncp: 
            best_ncp = total_ncp 
            best_config = (addition_set, remain_set, temp_less.middle, temp_more.middle) 
     
    # Second method - merge partitions
    second_ncp, second_mid = NCP_dis(less.middle, more.middle) 
    second_ncp *= all_length 
     
    if best_config and best_ncp < second_ncp: 
        # First method is better - move records
        addition_set, remain_set, first_mid, r_middle = best_config 
        less.member.extend(addition_set) 
        less.middle = first_mid 
        more.member = remain_set 
        more.middle = r_middle 
        sub_partitions.append(more) 
    else: 
        # Second method is better - merge partitions
        less.member.extend(more.member) 
        less.middle = second_mid 
        less.can_split = False 
        
    sub_partitions.append(less) 
 
def can_split(partition): 
    """ 
    check if partition can be further splited. 
    """ 
    if partition.can_split is False: 
        return False 
    if len(partition) < 2 * GL_K: 
        return False 
    return True 
 
def anonymize(partition): 
    """ 
    Main procedure of top_down_greedy_anonymization. 
    recursively partition groups until not allowable. 
    """ 
    if can_split(partition) is False: 
        RESULT.append(partition) 
        return 
         
    u, v = get_pair(partition) 
    sub_partitions = distribute_record(u, v, partition) 
     
    # Check if splitting improves NCP before proceeding 
    parent_ncp = len(partition) * NCP(partition.middle) 
    child_ncp = sum(len(p) * NCP(p.middle) for p in sub_partitions) 
     
    # Only proceed with split if it improves NCP or the difference is minimal 
    if child_ncp > parent_ncp * 1.05:  # Allow 5% tolerance 
        RESULT.append(partition) 
        return 
     
    # Handle partitions that are too small 
    if len(sub_partitions[0]) < GL_K: 
        balance(sub_partitions, 0) 
    elif len(sub_partitions[1]) < GL_K: 
        balance(sub_partitions, 1) 
     
    # Verify partition integrity 
    p_sum = len(partition) 
    c_sum = sum(len(p) for p in sub_partitions) 
    if p_sum != c_sum: 
        # This should never happen but is a safeguard 
        RESULT.append(partition) 
        return 
         
    # Continue recursively 
    for sub_partition in sub_partitions: 
        anonymize(sub_partition) 
 
def init(att_trees, data, k, QI_num=-1): 
    """ 
    reset all global variables 
    """ 
    global GL_K, RESULT, QI_LEN, ATT_TREES, QI_RANGE, IS_CAT 
    ATT_TREES = att_trees 
    
    if QI_num <= 0: 
        QI_LEN = len(data[0]) - 1 
    else: 
        QI_LEN = QI_num 
        
    GL_K = k 
    RESULT = [] 
    QI_RANGE = [] 
    IS_CAT = [] 
    
    # Properly initialize attribute types and ranges
    for i in range(QI_LEN): 
        if isinstance(att_trees[i], NumRange): 
            IS_CAT.append(False) 
            # For numerical attributes, store the range
            range_values = att_trees[i].sort_value
            if range_values and len(range_values) > 1:
                QI_RANGE.append(float(range_values[-1]) - float(range_values[0]))
            else:
                QI_RANGE.append(1.0)  # Default range if no values
        else: 
            IS_CAT.append(True) 
            # For categorical attributes, use tree depth
            if isinstance(att_trees[i], dict) and '*' in att_trees[i]:
                QI_RANGE.append(len(att_trees[i]['*']))
            else:
                QI_RANGE.append(1.0)  # Default if no '*' key
 
def Top_Down_Greedy_Anonymization(att_trees, data, k, QI_num=-1): 
    """ 
    Top Down Greedy Anonymization algorithm for relational dataset 
    with numeric and categorical attributes 
    """ 
    init(att_trees, data, k, QI_num) 
    result = [] 
    middle = [] 
     
    for i in range(QI_LEN): 
        if IS_CAT[i]: 
            # For categorical attributes, start with the root
            middle.append('*') 
        else: 
            # For numerical attributes, use the range
            if hasattr(att_trees[i], 'value'): 
                middle.append(att_trees[i].value) 
            elif hasattr(att_trees[i], 'sort_value') and att_trees[i].sort_value: 
                # If it's a NumRange object with values
                min_val = att_trees[i].sort_value[0]
                max_val = att_trees[i].sort_value[-1]
                middle.append(f"{min_val},{max_val}")
            else:
                # Default value
                middle.append('0,1')
 
    whole_partition = Partition(data, middle) 
    start_time = time.time() 
    anonymize(whole_partition) 
    rtime = float(time.time() - start_time) 
     
    # Calculate NCP 
    ncp = 0.0 
    dp = 0.0 
    for sub_partition in RESULT: 
        gen_result = sub_partition.middle 
        rncp = NCP(gen_result) 
        for _ in range(len(sub_partition)): 
            result.append(gen_result[:]) 
        rncp *= len(sub_partition) 
        dp += len(sub_partition) ** 2 
        ncp += rncp 
        
    # Convert NCP to percentage 
    if len(data) > 0 and QI_LEN > 0:
        ncp /= len(data) 
        ncp /= QI_LEN 
        ncp *= 100 
 
    if __DEBUG:  
        print(f"Discernability Penalty={dp:.2E}") 
        print(f"K={k}") 
        print("Size of partitions:", len(RESULT)) 
        print([len(partition) for partition in RESULT]) 
        print(f"NCP = {ncp:.2f}%") 
        print(f"Total running time = {rtime:.2f}") 
 
    return (result, (ncp, rtime)) 