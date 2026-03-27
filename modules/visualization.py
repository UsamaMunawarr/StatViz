import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import scipy.stats as stats
import plotly.graph_objects as go

def run():
    st.header("📈 Visualization & Plot Builder")
    if st.session_state.data is not None:
        df = st.session_state.data
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Plot Settings")
            chart_type = st.selectbox("Chart Type", [
                "Scatter Plot", "Bar Chart", "Line Chart", "Box Plot", 
                "Violin Plot", "Histogram", "Pie Chart", "Heatmap", 
                "Area Chart", "Bubble Chart", "Pair Plot", "QQ Plot", "Count Plot"
            ])
            
            x_col = st.selectbox("X-Axis Variable", ["None"] + list(df.columns))
            y_col = st.selectbox("Y-Axis Variable", ["None"] + list(df.columns))
            color_col = st.selectbox("Group By / Color Variable", ["None"] + list(df.columns))
            
            if chart_type == "Bubble Chart":
                size_col = st.selectbox("Size Variable (Required for Bubble)", df.select_dtypes(include=np.number).columns)
            else:
                size_col = None
        
        with col2:
            st.subheader(f"{chart_type} Preview")
            fig = None
            
            try:
                # Handle plot generation based on chart type
                x_val = x_col if x_col != "None" else None
                y_val = y_col if y_col != "None" else None
                color_val = color_col if color_col != "None" else None
                
                if chart_type == "Scatter Plot" and x_val and y_val:
                    fig = px.scatter(df, x=x_val, y=y_val, color=color_val, trendline="ols")
                elif chart_type == "Bar Chart" and x_val and y_val:
                    fig = px.bar(df, x=x_val, y=y_val, color=color_val, barmode="group")
                elif chart_type == "Line Chart" and x_val and y_val:
                    fig = px.line(df, x=x_val, y=y_val, color=color_val)
                elif chart_type == "Box Plot" and y_val:
                    fig = px.box(df, x=x_val, y=y_val, color=color_val)
                elif chart_type == "Violin Plot" and y_val:
                    fig = px.violin(df, x=x_val, y=y_val, color=color_val, box=True)
                elif chart_type == "Histogram" and x_val:
                    fig = px.histogram(df, x=x_val, color=color_val, marginal="box")
                elif chart_type == "Pie Chart" and x_val and y_val:
                    fig = px.pie(df, names=x_val, values=y_val, color=color_val)
                elif chart_type == "Heatmap":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) >= 2:
                        corr = df[numeric_cols].corr()
                        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
                    else:
                        st.info("Need at least 2 numeric columns for Heatmap.")
                elif chart_type == "Area Chart" and x_val and y_val:
                    fig = px.area(df, x=x_val, y=y_val, color=color_val)
                elif chart_type == "Bubble Chart" and x_val and y_val and size_col:
                    fig = px.scatter(df, x=x_val, y=y_val, color=color_val, size=size_col)
                elif chart_type == "Pair Plot":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) >= 2:
                        fig = px.scatter_matrix(df, dimensions=numeric_cols, color=color_val)
                elif chart_type == "QQ Plot" and y_val:
                    # Generate QQ plot using statsmodels/scipy logic manually with Plotly
                    numeric_data = df[y_val].dropna()
                    osm, osr = stats.probplot(numeric_data, dist="norm", fit=False)
                    fig = px.scatter(x=osm, y=osr, labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'})
                    # Add reference line
                    slope, intercept, _, _, _ = stats.linregress(osm, osr)
                    fig.add_trace(go.Scatter(x=osm, y=slope*osm + intercept, mode='lines', name='Normal Line'))
                elif chart_type == "Count Plot" and x_val:
                    fig = px.histogram(df, x=x_val, color=color_val, histfunc="count")
                else:
                    st.info(f"Please select appropriate X and/or Y variables for '{chart_type}'.")
                
                if fig:
                    fig.update_layout(template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly", margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error rendering plot: {e}")

        # Central Export Settings at the bottom
        if fig:
            st.divider()
            st.markdown("<h3 style='text-align: center'>Export Settings</h3>", unsafe_allow_html=True)
            
            col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
            
            with col_exp1:
                export_format = st.selectbox("Format", ["png", "jpeg", "svg"])
            with col_exp2:
                width = st.number_input("Width (px)", min_value=100, max_value=4000, value=800)
            with col_exp3:
                height = st.number_input("Height (px)", min_value=100, max_value=4000, value=600)
            with col_exp4:
                dpi = st.selectbox("DPI (Resolution)", [72, 150, 300])

            st.write("") # Spacing
            
            # Export Action
            try:
                scale_factor = dpi / 72.0
                fig.update_layout(width=width, height=height)
                img_bytes = fig.to_image(format=export_format, width=width, height=height, scale=scale_factor)
                
                st.download_button(
                    label=f"Download {export_format.upper()} (DPI: {dpi})",
                    data=img_bytes,
                    file_name=f"plot.{export_format}",
                    mime=f"image/{export_format}",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Image export error (requires kaleido): {e}")

    else:
        st.warning("Please upload data first in the Data Import module.")
