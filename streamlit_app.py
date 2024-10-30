# interface/streamlit_app.py

import streamlit as st
import asyncio
from typing import Dict
import logging
from datetime import datetime

from enhanced_learning_system import EnhancedLearningSystem

def create_interface():
    st.title("Alice - Enhanced Learning System")
    
    # Initialize system
    if 'system' not in st.session_state:
        st.session_state.system = EnhancedLearningSystem()
    
    # System state display
    st.sidebar.header("System State")
    system_state = st.session_state.system.system_state.get_summary()
    st.sidebar.json(system_state)
    
    # Quantum state display
    st.sidebar.header("Quantum State")
    quantum_state = st.session_state.system.consciousness.quantum_state
    st.sidebar.json(quantum_state)
    
    # Task input
    st.header("Task Input")
    task = st.text_area("Enter task description:")
    
    if st.button("Process Task"):
        if task:
            with st.spinner("Processing task..."):
                results = asyncio.run(st.session_state.system.process_web_task(task))
                
                # Display results
                st.header("Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Web Interaction Results")
                    st.json(results['web_results'])
                    
                    st.subheader("Detected Patterns")
                    st.write(results['patterns'])
                    
                with col2:
                    st.subheader("Generated Rules")
                    for rule in results['rules']:
                        st.write(f"- {rule}")
                    
                    st.subheader("System Analysis")
                    st.write(results['analysis'])
                
                # Display complexity layers
                st.header("Complexity Analysis")
                for layer in st.session_state.system.complexity_system.layers:
                    st.subheader(f"Layer: {layer.layer_type}")
                    st.write(f"Complexity Score: {layer.complexity_score}")
                    st.write(f"Emergent Properties: {layer.emergent_properties}")
        else:
            st.warning("Please enter a task description.")


