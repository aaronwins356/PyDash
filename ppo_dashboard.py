"""
PPO (Proximal Policy Optimization) Dashboard
==============================================
This Streamlit dashboard visualizes the mathematics behind PPO's clipping mechanism,
demonstrating how it constrains policy updates to ensure stable learning.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# COMPUTATION FUNCTIONS
# ============================================================================

def compute_unclipped_objective(r_theta, advantage):
    """
    Compute the unclipped policy gradient objective.
    
    Parameters:
    -----------
    r_theta : float or np.ndarray
        Policy ratio: pi_theta(a|s) / pi_old(a|s)
    advantage : float
        Advantage estimate A(s,a)
    
    Returns:
    --------
    float or np.ndarray
        Unclipped objective value: r(theta) * A
    """
    return r_theta * advantage


def compute_clipped_objective(r_theta, advantage, epsilon):
    """
    Compute the clipped surrogate objective used in PPO.
    
    Parameters:
    -----------
    r_theta : float or np.ndarray
        Policy ratio: pi_theta(a|s) / pi_old(a|s)
    advantage : float
        Advantage estimate A(s,a)
    epsilon : float
        Clipping parameter (typically 0.1 to 0.3)
    
    Returns:
    --------
    float or np.ndarray
        Clipped objective: min(r*A, clip(r, 1-eps, 1+eps)*A)
    """
    # Clip the ratio to [1-epsilon, 1+epsilon]
    r_clipped = np.clip(r_theta, 1 - epsilon, 1 + epsilon)
    
    # Take the minimum of clipped and unclipped objectives
    objective_unclipped = r_theta * advantage
    objective_clipped = r_clipped * advantage
    
    return np.minimum(objective_unclipped, objective_clipped)


def identify_clipping_regions(r_theta, advantage, epsilon):
    """
    Identify where clipping is active.
    
    Parameters:
    -----------
    r_theta : np.ndarray
        Policy ratio values
    advantage : float
        Advantage estimate
    epsilon : float
        Clipping parameter
    
    Returns:
    --------
    tuple of np.ndarray
        (lower_clip_mask, upper_clip_mask) - boolean arrays indicating clipping regions
    """
    if advantage >= 0:
        # For positive advantage, clipping occurs when r > 1 + epsilon
        lower_clip = np.zeros_like(r_theta, dtype=bool)
        upper_clip = r_theta > (1 + epsilon)
    else:
        # For negative advantage, clipping occurs when r < 1 - epsilon
        lower_clip = r_theta < (1 - epsilon)
        upper_clip = np.zeros_like(r_theta, dtype=bool)
    
    return lower_clip, upper_clip


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_ppo_objectives(r_theta_range, advantage, epsilon):
    """
    Create a matplotlib figure showing both clipped and unclipped objectives.
    
    Parameters:
    -----------
    r_theta_range : tuple
        (min, max) values for r(theta) to plot
    advantage : float
        Advantage estimate A(s,a)
    epsilon : float
        Clipping parameter
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Generate r(theta) values
    r_theta = np.linspace(r_theta_range[0], r_theta_range[1], 1000)
    
    # Compute objectives
    unclipped = compute_unclipped_objective(r_theta, advantage)
    clipped = compute_clipped_objective(r_theta, advantage, epsilon)
    
    # Identify clipping regions
    lower_clip, upper_clip = identify_clipping_regions(r_theta, advantage, epsilon)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot unclipped objective
    ax.plot(r_theta, unclipped, 'b--', linewidth=2, label='Unclipped: r(θ)·A', alpha=0.7)
    
    # Plot clipped objective
    ax.plot(r_theta, clipped, 'r-', linewidth=2.5, label='Clipped (PPO): min(r·A, clip(r)·A)')
    
    # Highlight clipping regions
    if np.any(lower_clip):
        ax.fill_between(r_theta, np.min(clipped), np.max(unclipped), 
                        where=lower_clip, alpha=0.2, color='orange',
                        label=f'Lower clip region (r < {1-epsilon:.2f})')
    
    if np.any(upper_clip):
        ax.fill_between(r_theta, np.min(clipped), np.max(unclipped),
                        where=upper_clip, alpha=0.2, color='red',
                        label=f'Upper clip region (r > {1+epsilon:.2f})')
    
    # Mark the original policy point (r=1)
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Old policy (r=1)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Mark clipping boundaries
    ax.axvline(x=1-epsilon, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=1+epsilon, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('Policy Ratio r(θ) = π_θ(a|s) / π_old(a|s)', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title(f'PPO Clipping Mechanism (ε={epsilon}, A={advantage:.2f})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="PPO Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    # Title and introduction
    st.title("🤖 Proximal Policy Optimization (PPO) Dashboard")
    st.markdown("**Interactive visualization of PPO's clipping mechanism**")
    
    # Sidebar controls
    st.sidebar.header("📊 Control Parameters")
    st.sidebar.markdown("---")
    
    # Epsilon (clipping parameter)
    epsilon = st.sidebar.slider(
        "Clipping Parameter (ε)",
        min_value=0.01,
        max_value=0.5,
        value=0.2,
        step=0.01,
        help="Controls how much the policy can change. Typical values: 0.1-0.3"
    )
    
    # Advantage
    advantage = st.sidebar.slider(
        "Advantage A(s,a)",
        min_value=-5.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Positive advantage = good action, Negative = bad action"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Policy Ratio Range")
    
    # r(theta) range
    r_min = st.sidebar.number_input(
        "Minimum r(θ)",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1
    )
    
    r_max = st.sidebar.number_input(
        "Maximum r(θ)",
        min_value=0.5,
        max_value=3.0,
        value=2.0,
        step=0.1
    )
    
    # Ensure valid range
    if r_min >= r_max:
        st.sidebar.error("⚠️ Minimum must be less than Maximum!")
        return
    
    # Main visualization
    st.markdown("---")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Objective Function Visualization")
        
        # Generate and display plot
        fig = plot_ppo_objectives((r_min, r_max), advantage, epsilon)
        st.pyplot(fig)
        plt.close(fig)  # Clean up
    
    with col2:
        st.subheader("📋 Current Values")
        st.metric("Epsilon (ε)", f"{epsilon:.3f}")
        st.metric("Advantage (A)", f"{advantage:.2f}")
        st.metric("Clip Range", f"[{1-epsilon:.3f}, {1+epsilon:.3f}]")
        
        st.markdown("---")
        
        # Show what happens at key points
        st.markdown("**At r(θ) = 1.0:**")
        obj_at_1 = compute_clipped_objective(1.0, advantage, epsilon)
        st.write(f"Objective = {obj_at_1:.3f}")
        
        st.markdown(f"**At r(θ) = {1+epsilon:.3f} (upper clip):**")
        obj_at_upper = compute_clipped_objective(1+epsilon, advantage, epsilon)
        st.write(f"Objective = {obj_at_upper:.3f}")
        
        st.markdown(f"**At r(θ) = {1-epsilon:.3f} (lower clip):**")
        obj_at_lower = compute_clipped_objective(1-epsilon, advantage, epsilon)
        st.write(f"Objective = {obj_at_lower:.3f}")
    
    # Educational content
    st.markdown("---")
    st.header("📚 Understanding PPO")
    
    # Create tabs for different explanations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Why Clipping?", 
        "Policy Ratio r(θ)", 
        "Advantages", 
        "The Math"
    ])
    
    with tab1:
        st.markdown("""
        ### Why Does PPO Use Clipping?
        
        **Problem:** Standard policy gradient methods can make large, destabilizing updates that 
        hurt performance.
        
        **Solution:** PPO clips the objective function to prevent excessively large policy updates.
        
        #### How It Works:
        1. **Positive Advantage (good action):** 
           - We want to increase its probability
           - But clipping at `1+ε` prevents making it *too* likely (over-optimization)
        
        2. **Negative Advantage (bad action):**
           - We want to decrease its probability
           - But clipping at `1-ε` prevents making it *too* unlikely (over-correction)
        
        #### Key Benefits:
        - ✅ **Stability:** Prevents catastrophic policy collapses
        - ✅ **Sample efficiency:** Can reuse data for multiple epochs
        - ✅ **Simplicity:** No complex KL penalty calculations needed
        - ✅ **Performance:** Works reliably across many environments
        
        **Try it:** Set advantage to +2.0 and watch how the objective plateaus beyond `1+ε`!
        """)
    
    with tab2:
        st.markdown("""
        ### Understanding the Policy Ratio r(θ)
        
        The policy ratio measures how much the new policy differs from the old one:
        
        ```
        r(θ) = π_θ(a|s) / π_old(a|s)
        ```
        
        #### Interpretation:
        - **r(θ) = 1.0**: New policy exactly matches old policy (no change)
        - **r(θ) > 1.0**: New policy assigns *higher* probability to action `a`
        - **r(θ) < 1.0**: New policy assigns *lower* probability to action `a`
        
        #### Examples:
        - `r(θ) = 2.0` → New policy is **2x** more likely to take action `a`
        - `r(θ) = 0.5` → New policy is **2x** less likely to take action `a`
        
        #### Why It Matters:
        The ratio tells us *how much* we're changing the policy. Without clipping, 
        large ratios (like r=5.0) could lead to drastic, harmful changes.
        
        **PPO's guarantee:** The ratio stays within `[1-ε, 1+ε]` for the optimization,
        limiting the policy change to a safe range.
        """)
    
    with tab3:
        st.markdown("""
        ### How Advantages Shape the Curve
        
        The advantage function `A(s,a)` estimates how much better action `a` is compared 
        to the average action in state `s`.
        
        #### Sign and Magnitude:
        - **A > 0**: Action is better than average → *increase* its probability
        - **A < 0**: Action is worse than average → *decrease* its probability
        - **|A| large**: Strong signal → steeper objective curve
        - **|A| small**: Weak signal → flatter objective curve
        
        #### Effect on Clipping:
        
        **Positive Advantage (A > 0):**
        - Unclipped objective grows with r(θ)
        - Clipping activates at upper bound (r > 1+ε)
        - Prevents *over-encouraging* good actions
        
        **Negative Advantage (A < 0):**
        - Unclipped objective decreases with r(θ)
        - Clipping activates at lower bound (r < 1-ε)
        - Prevents *over-discouraging* bad actions
        
        **Interactive Experiment:**
        1. Set A = +3.0 (very good action) → See upper clipping
        2. Set A = -3.0 (very bad action) → See lower clipping
        3. Set A = 0.0 (neutral) → No preference, flat objective
        """)
    
    with tab4:
        st.markdown("""
        ### The Mathematics of PPO
        
        #### Standard Policy Gradient Objective:
        ```
        L^PG(θ) = 𝔼[r(θ) · A(s,a)]
        ```
        Problem: Can lead to excessively large updates.
        
        #### PPO Clipped Objective:
        ```
        L^CLIP(θ) = 𝔼[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]
        ```
        
        Where:
        - `r(θ) = π_θ(a|s) / π_old(a|s)` (probability ratio)
        - `A = A(s,a)` (advantage estimate)
        - `ε` (epsilon) is the clipping parameter (e.g., 0.2)
        - `clip(x, min, max)` constrains x to [min, max]
        
        #### Why the Minimum?
        
        The `min()` operation creates a **pessimistic bound**:
        
        1. **When advantage is positive (good action):**
           - If r(θ) > 1+ε: We cap the benefit at (1+ε)·A
           - Prevents over-optimization
        
        2. **When advantage is negative (bad action):**
           - If r(θ) < 1-ε: We cap the penalty at (1-ε)·A
           - Prevents over-penalization
        
        #### In Practice:
        PPO maximizes this objective over multiple epochs using the same batch of data,
        which is more sample-efficient than vanilla policy gradients.
        """)
    
    # Footer with extension ideas
    st.markdown("---")
    st.header("🔧 Extending This Dashboard")
    
    with st.expander("Click to see extension ideas"):
        st.markdown("""
        ### Potential Extensions:
        
        #### 1. **Add KL Divergence Penalty**
        - Show the alternative PPO formulation with adaptive KL penalties
        - Visualize: `L^KLPEN(θ) = 𝔼[r(θ)·A - β·KL(π_old||π_θ)]`
        - Add slider for β (KL coefficient)
        
        #### 2. **Entropy Bonus**
        - Encourage exploration with entropy term
        - Visualize: `L^TOTAL(θ) = L^CLIP(θ) + c₁·L^VF(θ) + c₂·S[π_θ]`
        - Add slider for c₂ (entropy coefficient)
        
        #### 3. **Value Function Component**
        - Show the value loss: `L^VF = (V_θ(s) - V^target)²`
        - Combined actor-critic visualization
        
        #### 4. **Multi-Action Comparison**
        - Compare clipping behavior for multiple actions simultaneously
        - Show how different advantages lead to different clipping patterns
        
        #### 5. **Training Trajectory Simulation**
        - Animate how policy ratio evolves over training epochs
        - Show convergence to optimal policy
        
        #### 6. **Real Environment Integration**
        - Connect to Gym environments (CartPole, LunarLander)
        - Live PPO training with real-time objective visualization
        
        #### 7. **Comparison with Other Methods**
        - Side-by-side with TRPO (Trust Region Policy Optimization)
        - Show vanilla policy gradient behavior without clipping
        
        #### 8. **3D Visualization**
        - Plot objective as function of both r(θ) and ε
        - Interactive 3D surface plot with Plotly
        
        #### 9. **Batch Statistics**
        - Simulate a batch of (s,a) pairs with different advantages
        - Show distribution of clipping across the batch
        
        #### 10. **Hyperparameter Sensitivity Analysis**
        - Automated sweep over ε values
        - Show impact on policy update magnitude
        
        ---
        
        ### Code Structure for Extensions:
        
        ```python
        # Add new computation functions
        def compute_kl_penalty(r_theta, beta):
            return beta * (r_theta * np.log(r_theta) - (r_theta - 1))
        
        # Add to plotting function
        def plot_with_kl(r_theta_range, advantage, epsilon, beta):
            # ... existing code ...
            kl_penalty = compute_kl_penalty(r_theta, beta)
            ax.plot(r_theta, clipped - kl_penalty, label='With KL penalty')
        
        # Add new sidebar control
        beta = st.sidebar.slider("KL Coefficient (β)", 0.0, 1.0, 0.01)
        ```
        
        Feel free to fork and extend this dashboard for your own research or learning!
        """)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()


# ============================================================================
# EXTENSION NOTES
# ============================================================================
# 
# To extend this dashboard:
# 1. Add new computation functions following the pattern above
# 2. Create corresponding plotting functions or modify existing ones
# 3. Add sidebar controls for new parameters
# 4. Add educational tabs explaining the new concepts
# 5. Keep the code modular and well-commented for easy modification
#
# Suggested improvements:
# - Add KL divergence penalty visualization
# - Include entropy bonus term
# - Show value function loss component
# - Add animation of policy updates over time
# - Connect to real RL environments for live training visualization
# - Add comparison with other policy gradient methods (TRPO, A3C, etc.)
# ============================================================================
