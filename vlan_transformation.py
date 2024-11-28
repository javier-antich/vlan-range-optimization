import streamlit as st
import os
import re
from collections import defaultdict
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
from typing import List, Tuple
import time
import threading
import queue
import sys
import multiprocessing
from streamlit_autorefresh import st_autorefresh
import subprocess



def parse_include(include: str) -> List[Tuple[int, int]]:
    """
    Parses a comma-separated string of ranges and discrete numbers into a list of tuples.

    Args:
        include (str): A string containing comma-separated ranges and/or discrete numbers.
                       Examples:
                           '1-1000,1002-1500,1631'
                           '5,10-20,25'

    Returns:
        List[Tuple[int, int]]: A list of tuples where each tuple represents a range.
                               For discrete numbers, the tuple contains the same number twice.
                               Example:
                                   [(1, 1000), (1002, 1500), (1631, 1631)]
    """
    result = []
    # Split the input string by commas to process each part
    parts = include.split(',')

    for part in parts:
        part = part.strip()  # Remove any leading/trailing whitespace
        if '-' in part:
            # It's a range; split by hyphen
            try:
                start_str, end_str = part.split('-', 1)
                start = int(start_str.strip())
                end = int(end_str.strip())+1
                if start > end:
                    raise ValueError(f"Invalid range '{part}': start > end.")
                result.append((start, end))
            except ValueError as ve:
                print(f"Skipping invalid range '{part}': {ve}")
        else:
            # It's a discrete number
            try:
                num = int(part)
                result.append((num, num+1))
            except ValueError as ve:
                print(f"Skipping invalid number '{part}': {ve}")

    return result

def parse_exclude(exclude_string):
    if len(exclude_string)>0:
        exclude_list = [int(i) for i in exclude_string.split(',')]
    else:
        exclude_list = []
    return exclude_list

def check_containment(k: int, intervals: List[Tuple[int, int]]) -> List[bool]:
    """
    Checks whether the integer k is contained in each interval.

    Parameters:
    - k (int): The integer to check.
    - intervals (List[Tuple[int, int]]): A list of tuples representing intervals (a, b).

    Returns:
    - List[bool]: A list where each element is True if k is in the corresponding interval, else False.
    """
    containment = []
    for index, (a, b) in enumerate(intervals):
        if a > b:
            raise ValueError(f"Invalid interval at index {index}: ({a}, {b}). Lower bound must be <= upper bound.")
        is_contained = a <= k < b
        containment.append(is_contained)
    return any(containment)

def check_discrete(k: int, intervals: List[Tuple[int, int]]) -> List[bool]:
    """
    Checks whether the integer k is contained in an interval of length 1 (i.e. is discrete).

    Parameters:
    - k (int): The integer to check.
    - intervals (List[Tuple[int, int]]): A list of tuples representing intervals (a, b).

    Returns:
    - bool: if k is part of a discrete item in the input intervals list.
    """
    containment = []
    for index, (a, b) in enumerate(intervals):
        if a > b:
            raise ValueError(f"Invalid interval at index {index}: ({a}, {b}). Lower bound must be <= upper bound.")
        is_discrete = (a <= k < b) and ((b-a)==1)
        containment.append(is_discrete)
    return any(containment)

def transform_vlans(include_list,exclude,num_ranges = 20):
    parsed_include = []
    for item in include_list:
        parsed_include.append(parse_include(item))
        #format of parsed_include is [(a,b),(c,d),(e,e+1),...]
    parsed_exclude = parse_exclude(exclude)
    #format of parsed_exclude is: [x,y,z,...]
    num_lines = len(parsed_include)
    print('Parsed include:',parsed_include)
    print('Parsed exclude:',parsed_exclude)
    print('Number of lines to process:',num_lines)
    model = cp_model.CpModel()

    #definition of the variables to compute. The start and end point of each vlan range for output
    output_ranges = []
    output_intervals = []
    line_active_vars = []
    for line in range(num_lines):
        line_output = []
        line_intervals = []
        
        for vlan_range in range(num_ranges):
            start = model.NewIntVar(1, 4095, f'start_{line}_{vlan_range}')
            length = model.NewIntVar(1, 4094, f'length_{line}_{vlan_range}')
            end = model.NewIntVar(1, 4095, f'end_{line}_{vlan_range}')
            active = model.NewBoolVar(f'active_{line}_{vlan_range}')
            # Link start, duration, and end
            model.Add(end == start + length).OnlyEnforceIf(active)
            model.Add(end == start).OnlyEnforceIf(active.Not())

            #here we set the condition that the range starts at least after where the previous range ends.
            if vlan_range>0:
                for previous_range in range(vlan_range):
                    model.Add(start >= line_output[previous_range][1]).OnlyEnforceIf(line_output[previous_range][3])
            line_active_vars.append(active)
            # Create interval variable
            #interval = model.NewOptionalIntervalVar(start, length, end, active, f'interval_{line}_{vlan_range}')
            
            line_output.append((start,end,length,active))
            #line_intervals.append(interval)

        output_ranges.append(line_output)
        #output_intervals.append(line_intervals)

    #start defining constraints
    #test = """
    #make sure intervals within a line do not overlap
    for line in range(num_lines):
        #model.AddNoOverlap(output_intervals[line])
        #test = """
        for interval in range(len(output_ranges[line])):

                    for t in range(len(output_ranges[line])):
                        total_overlaps = []
                        if not t==interval:
                            # Define Boolean variables for each condition
                            no_overlap_before = model.NewBoolVar(f'X{line}_{interval}_end_before_Y{line}_{t}_start')
                            no_overlap_after = model.NewBoolVar(f'X{line}_{interval}_start_after_Y{line}_{t}_end')
                            overlaps = model.NewBoolVar(f'X{line}_{interval}_overlaps_with_Y{line}_{t}')
                            both_active = model.NewBoolVar(f'X{line}_{interval}_both_active_Y{line}_{t}')
                            
                            
                            # Condition 1: X ends before Y starts -> it does not overlap
                            model.Add(output_ranges[line][interval][1] <= output_ranges[line][t][0]).OnlyEnforceIf(no_overlap_before)
                            model.Add(output_ranges[line][interval][1] > output_ranges[line][t][0]).OnlyEnforceIf(no_overlap_before.Not())
                            
                            # Condition 2: X starts after Y ends -> it does not overlap
                            model.Add(output_ranges[line][interval][0] >= output_ranges[line][t][1]).OnlyEnforceIf(no_overlap_after)
                            model.Add(output_ranges[line][interval][0] < output_ranges[line][t][1]).OnlyEnforceIf(no_overlap_after.Not())
                            
                            # Condition 3: this condition means it overlaps
                            model.Add(output_ranges[line][interval][0] >= output_ranges[line][t][0]).OnlyEnforceIf(overlaps)
                            model.Add(output_ranges[line][interval][1] <= output_ranges[line][t][1]).OnlyEnforceIf(overlaps)


                            # Ensure exactly one of the conditions holds
                            #model.AddBoolOr([no_overlap_before, no_overlap_after]).OnlyEnforceIf(overlaps.Not())
                            model.Add(no_overlap_before + no_overlap_after == 1).OnlyEnforceIf(overlaps.Not())
                            model.Add(overlaps == 0).OnlyEnforceIf([output_ranges[line][interval][3],output_ranges[line][t][3]])
                            total_overlaps.append(overlaps)

                    
    #"""

    #now we need to make sure that no range overlaps with any of the ranges in other lines, with the exception of being equal
    #"""
    #test = """
    for line in range(num_lines):
        for interval in range(num_ranges):
            for line2 in range(num_lines):
                if not line2 == line:
                    
                    for t in range(num_ranges):
                        # Define Boolean variables for each condition
                        no_overlap_before = model.NewBoolVar(f'X{line}_{interval}_end_before_Y{line2}_{t}_start')
                        no_overlap_after = model.NewBoolVar(f'X{line}_{interval}_start_after_Y{line2}_{t}_end')
                        overlaps = model.NewBoolVar(f'X{line}_{interval}_overlaps_with_Y{line}_{t}')
                        coincide = model.NewBoolVar(f'X{line}_{interval}_coincide_Y{line2}_{t}')
                        
                        # Condition 1: X ends before Y starts -> it does not overlap
                        model.Add(output_ranges[line][interval][1] <= output_ranges[line2][t][0]).OnlyEnforceIf(no_overlap_before)
                        model.Add(output_ranges[line][interval][1] > output_ranges[line2][t][0]).OnlyEnforceIf(no_overlap_before.Not())
                            
                            # Condition 2: X starts after Y ends -> it does not overlap
                        model.Add(output_ranges[line][interval][0] >= output_ranges[line2][t][1]).OnlyEnforceIf(no_overlap_after)
                        model.Add(output_ranges[line][interval][0] < output_ranges[line2][t][1]).OnlyEnforceIf(no_overlap_after.Not())
                            
                            # Condition 3: this condition means it overlaps
                        model.Add(output_ranges[line][interval][0] >= output_ranges[line2][t][0]).OnlyEnforceIf(overlaps)
                        model.Add(output_ranges[line][interval][1] <= output_ranges[line2][t][1]).OnlyEnforceIf(overlaps)
                        
                        # Condition 3: X coincides with Y
                        model.Add(output_ranges[line][interval][0] == output_ranges[line2][t][0]).OnlyEnforceIf(coincide)
                        model.Add(output_ranges[line][interval][1] == output_ranges[line2][t][1]).OnlyEnforceIf(coincide)
                        
                        # Ensure exactly one of the conditions holds
                        model.AddBoolOr([no_overlap_before, no_overlap_after, coincide]).OnlyEnforceIf([output_ranges[line][interval][3],output_ranges[line2][t][3]])
                        model.Add(no_overlap_before + no_overlap_after + coincide == 1).OnlyEnforceIf([output_ranges[line][interval][3],output_ranges[line2][t][3]])
                        model.Add(overlaps == 0).OnlyEnforceIf([output_ranges[line][interval][3],output_ranges[line][t][3]])
    #"""
    #test="""
    #now we need to make sure that out of the exclude lists, no interval contains this number unless it is a interval of length 1
    for k in parsed_exclude:
        for line in range(num_lines):
            for t, interval in enumerate(output_ranges[line]):

                # Define a Boolean variable indicating if interval L[t] contains k
                contains = model.NewBoolVar(f'contains_k{k}_L{line}_{t}')
                is_below = model.NewBoolVar('is_below_k{k}_L{line}_{t}')
                is_above = model.NewBoolVar('is_above_k{k}_L{line}_{t}')
                exactly = model.NewBoolVar(f'exactly_m{k}_interval_{line}_{t}')
                    
                    # Add constraints to define when 'contains' is True
                    # a <= k < b
                model.Add(output_ranges[line][t][0] <= k).OnlyEnforceIf(contains)
                model.Add(output_ranges[line][t][1] > k).OnlyEnforceIf(contains)

                model.Add(output_ranges[line][t][0] > k).OnlyEnforceIf(is_above)
                model.Add(output_ranges[line][t][1] <= k).OnlyEnforceIf(is_below)

                model.Add(output_ranges[line][t][0] <= k).OnlyEnforceIf(is_above.Not())
                model.Add(output_ranges[line][t][1] > k).OnlyEnforceIf(is_below.Not())

                    #model.AddBoolOr([is_below, is_above]).OnlyEnforceIf(contains.Not())
                model.Add(is_below + is_above == 1).OnlyEnforceIf(contains.Not())
                

                # Condition 2: Interval is exactly [m, m+1)
                model.Add(interval[0] == k).OnlyEnforceIf(exactly)
                model.Add(interval[1] == k + 1).OnlyEnforceIf(exactly)
                model.Add(interval[2] == 1).OnlyEnforceIf(exactly)

                model.Add(exactly == 1).OnlyEnforceIf(contains)

                
    #"""
    #now we need to make sure that all vlans contained in the include list are also included in the output ranges.
    #test="""
    for k in range(4095):
        # Enforce that each k in K is contained in exactly one interval in L
        for line in range(num_lines):
            if check_containment(k,parsed_include[line]):
    
                contains_vars = []
                active_vars = []
                for t in range(num_ranges):
                    # Define a Boolean variable indicating if interval L[t] contains k
                    contains = model.NewBoolVar(f'contains_k{k}_L{line}_{t}')
                    is_below = model.NewBoolVar('is_below_k{k}_L{line}_{t}')
                    is_above = model.NewBoolVar('is_above_k{k}_L{line}_{t}')
                    
                    # Add constraints to define when 'contains' is True
                    # a <= k < b
                    model.Add(output_ranges[line][t][0] <= k).OnlyEnforceIf(contains)
                    model.Add(output_ranges[line][t][1] > k).OnlyEnforceIf(contains)

                    model.Add(output_ranges[line][t][0] > k).OnlyEnforceIf(is_above)
                    model.Add(output_ranges[line][t][1] <= k).OnlyEnforceIf(is_below)

                    model.Add(output_ranges[line][t][0] <= k).OnlyEnforceIf(is_above.Not())
                    model.Add(output_ranges[line][t][1] > k).OnlyEnforceIf(is_below.Not())

                    #model.AddBoolOr([is_below, is_above]).OnlyEnforceIf(contains.Not())
                    model.Add(is_below + is_above == 1).OnlyEnforceIf(contains.Not())
                    
                    #model.Add(contains == 1).OnlyEnforceIf(output_ranges[line][t][3])
                    #model.Add(contains == 0).OnlyEnforceIf(output_ranges[line][t][3].Not())
                    active_vars.append(output_ranges[line][t][3])
                    contains_vars.append(contains)
                    # Enforce that the interval containing s has duration ==1
                    if check_discrete(k,parsed_include[line]):
                        # If interval L[t] contains s, then duration_L[t] ==1
                        model.Add(output_ranges[line][t][2] == 1).OnlyEnforceIf(contains)
                        # If interval L[t] does not contain s, no constraint on duration
                # Enforce that exactly one interval contains k
                s_i_list=[]
                for i in range(len(contains_vars)):
                    l_i = contains_vars[i]
                    m_i = active_vars[i]

                    # Define the intermediate Boolean variable s_i = l_i AND m_i
                    s_i = model.NewBoolVar(f's_{i}')
                    s_i_list.append(s_i)

                    # Link s_i to l_i and m_i
                    model.AddBoolAnd([l_i, m_i]).OnlyEnforceIf(s_i)
                    model.AddBoolOr([l_i.Not(), m_i.Not()]).OnlyEnforceIf(s_i.Not())

                # Enforce that exactly one s_i is True
                model.Add(sum(s_i_list) == 1)
                
            else:
                contains_vars = []
                active_vars = []
                for t in range(num_ranges):
                    # Define a Boolean variable indicating if interval L[t] contains k
                    contains = model.NewBoolVar(f'contains_k{k}_L{line}_{t}')
                    is_below = model.NewBoolVar('is_below_k{k}_L{line}_{t}')
                    is_above = model.NewBoolVar('is_above_k{k}_L{line}_{t}')
                    
                    # Add constraints to define when 'contains' is True
                    # a <= k < b
                    model.Add(output_ranges[line][t][0] <= k).OnlyEnforceIf(contains)
                    model.Add(output_ranges[line][t][1] > k).OnlyEnforceIf(contains)

                    model.Add(output_ranges[line][t][0] > k).OnlyEnforceIf(is_above)
                    model.Add(output_ranges[line][t][1] <= k).OnlyEnforceIf(is_below)

                    model.Add(output_ranges[line][t][0] <= k).OnlyEnforceIf(is_above.Not())
                    model.Add(output_ranges[line][t][1] > k).OnlyEnforceIf(is_below.Not())

                    #model.AddBoolOr([is_below, is_above]).OnlyEnforceIf(contains.Not())
                    model.Add(is_below + is_above == 1).OnlyEnforceIf(contains.Not())

                    #model.Add(contains == 0).OnlyEnforceIf(output_ranges[line][t][3])
                    #model.Add(contains == 0).OnlyEnforceIf(output_ranges[line][t][3].Not())
                    active_vars.append(output_ranges[line][t][3])
                    contains_vars.append(contains)
                    

                # Enforce that exactly no interval contains k, but we only care about this for active intervals
                #model.Add(sum(contains_vars) == 1)
                for i in range(len(contains_vars)):
                    model.Add(contains_vars[i] == 0).OnlyEnforceIf(active_vars[i])    

          


    #"""
    model.Minimize(sum(line_active_vars))
            
    # Create the solver and solve
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True

    # Optionally, set a time limit (in seconds)
    #solver.parameters.max_time_in_seconds = 10.0

    # Solve the model
    status = solver.Solve(model)
    result = dict()
    result['solver_status']=str(solver.StatusName(status))
    #print(f'Solver status: {solver.StatusName(status)}')
    # Display the results
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        #print(f"Solution Found ")
        solution_list = []
        for line in range(num_lines):
            #print(f"For line {line}:")
            vlan_string = ''
            for vrange in range(len(output_ranges[line])):
                if solver.Value(output_ranges[line][vrange][3]):
                #if True:
                    a = solver.Value(output_ranges[line][vrange][3])
                    s = solver.Value(output_ranges[line][vrange][0])
                    e = solver.Value(output_ranges[line][vrange][1])
                    if (e-s)==1:
                        vlan_string+=str(s)+','
                    else:
                        vlan_string+=str(s)+'-'+str(e-1)+','
            solution_list.append(vlan_string)
            #print(f"Computed Vlan ranges: {vlan_string}")
        result['solution_list']=solution_list
                 
    else:
        #print("No solution found.")
        result['solution_list']='No solution found'
    return result

# -------------------- Processing Function --------------------

def processing_function(num_ranges, include_list, exclude_list, result_container):
    """
    Processes the input parameters using OR-Tools CP-SAT solver and updates the result.

    
    """
    
    #call vlan_transformation function
    result = transform_vlans(include_list,exclude_list,num_ranges)
    
    # Update the result in the shared container
    result_container['result'] = result

# Initialize Streamlit session state variables
if 'result' not in st.session_state:
    st.session_state.result = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'thread' not in st.session_state:
    st.session_state.thread = None
if 'result_container' not in st.session_state:
    st.session_state.result_container = {}

# -------------------- User Interface --------------------

# Title Banner with Vertical and Horizontal Centering
st.markdown(
    """
    <div style="
        background-color: black;
        padding: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100px;
    ">
        <h1 style="color: white; margin: 0;">VLAN Ranges optimization tool</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)  # Add a line break for spacing

# Parameters Introduction Section
st.subheader("Parameters introduction")

# -------------------- Dynamic Input for Parameter 2 --------------------


    # Parameter 1: Integer input
num_ranges = st.number_input(
        label="Max number of output VLAN ranges per line (Integer)",
        min_value=0,
        value=10,
        step=1,
        help="The algorithm needs a maximum number. The larger the more processing requires, but a small number may not be sufficient. 20 recommended."
    )

# Input for the number of strings in Parameter 2
num_strings = st.number_input(
    label="Number of subinterfaces with vlan ranges to optimize",
    min_value=0,
    max_value=20,
    value=1,
    step=1,
    help="One line of vlan ranges will need to be provided per subinterface"
)

# Dynamically generate text inputs based on the number selected
include_list = []
if num_strings > 0:
    for i in range(1, int(num_strings) + 1):
        string_input = st.text_input(
            label=f"Subinterface {i}",
            key=f'line_{i}',
            help=f"Enter the vlan-ranges for subinterface {i}. Example: 100-250,253,1000-1010"
        )
        include_list.append(string_input)
else:
    st.info("No vlan ranges provided")




    
    # Parameter 3: String input
exclude_list = st.text_input(
        label="VLANs to be excluded from ranges",
        value="",
        help="Enter list of vlans that should be excluded from any range. Example: 10,25,101,2050 "
    )
    
    # Submit button
submit_button = st.button(label='Process Transformation Request')

# Placeholder for spinner and timer
spinner_placeholder = st.empty()
timer_placeholder = st.empty()

# Placeholder for results
result_placeholder = st.empty()

# Handle form submission
if submit_button and not st.session_state.processing:
    # Reset previous result
    st.session_state.result = None
    st.session_state.result_container = {}
    
    # Record the start time
    st.session_state.start_time = time.time()
    
    # Set processing flag
    st.session_state.processing = True
    
    # Start the processing function in a separate thread
    st.session_state.thread = threading.Thread(
        target=processing_function,
        args=(num_ranges, include_list, exclude_list, st.session_state.result_container)
    )
    st.session_state.thread.start()

# If processing is ongoing, display spinner and timer
if st.session_state.processing:
    with spinner_placeholder.container():
        with st.spinner('Processing Transformation Request...'):
            pass  # Spinner is active

    # Calculate elapsed time
    elapsed_time = time.time() - st.session_state.start_time
    elapsed_display = f"Elapsed Time: {elapsed_time:.2f} seconds"
    timer_placeholder.text(elapsed_display)

    # Continuously update the timer until processing is complete
    while st.session_state.processing:
        time.sleep(0.5)  # Update every 0.5 seconds
        elapsed_time = time.time() - st.session_state.start_time
        elapsed_display = f"Elapsed Time: {elapsed_time:.2f} seconds"
        timer_placeholder.text(elapsed_display)
        
        # Check if the thread has finished
        if not st.session_state.thread.is_alive():
            st.session_state.processing = False
            st.session_state.result = st.session_state.result_container.get('result', "No result returned.")
            break

# Display the result if available
if st.session_state.result:
    st.subheader("Results:")
    st.write(st.session_state.result)

if __name__ == "__main__":
    #if len(sys.argv) > 1 and sys.argv[1] == 'run-solver':
        # Run the solver subprocess
    #    try:
    #        param1 = int(sys.argv[2])
    #        param2 = sys.argv[3]
    #        param3 = sys.argv[4]
    #        log_file_path = sys.argv[5]
    #    except (IndexError, ValueError):
    #        print("Invalid parameters", flush=True)
    #        sys.exit(1)
    #    run_solver(param1, param2, param3, log_file_path)
    #else:
        # Run the Streamlit application
    #    main_app()



    include_list = [
        '500-700,702-1143,1145-1267',
        '400-4000',
        
        

    ]
    exclude_list = '1852,1924,1925,3156,3157,4000'

    #transform_vlans(include_list,exclude_list,num_ranges = 20)



        

