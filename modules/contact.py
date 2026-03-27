import streamlit as st
import os
import base64

def run():
    st.header("📞 Contact & Developer Info")
    st.write("")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        img_path = "developer.jpeg"
        if os.path.exists(img_path):
            try:
                img_bytes = open(img_path, "rb").read()
                encoded = base64.b64encode(img_bytes).decode()
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: center; align-items: center; margin-top: 10px;">
                        <img src="data:image/jpeg;base64,{encoded}" 
                             style="border-radius: 50%; max-width: 260px; width: 100%; aspect-ratio: 1/1; 
                                    border: 4px solid #00d4aa; object-fit: cover; 
                                    box-shadow: 0 10px 25px rgba(0,212,170,0.3); transition: transform 0.3s ease;"
                             onmouseover="this.style.transform='scale(1.05)'" 
                             onmouseout="this.style.transform='scale(1)'">
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            except Exception:
                st.image(img_path, use_container_width=True)
        else:
            st.info("developer.jpeg image not found in the main folder")
            
    with col2:
        st.write("##### About the author:")
        st.markdown("<h1 style='color:#00d4aa; font-weight: 800; font-size: 55px; margin-bottom: 0px; padding-bottom: 0px; line-height: 1.1;'>Usama Munawar</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 18px; color: #a9a9a9; margin-top: 10px;'>Passionate Data Scientist and Developer building tools to make statistics and machine learning beautifully accessible to everyone.</p>", unsafe_allow_html=True)
        
        st.write("##### Connect with me:")
        
        # Define external link icons
        portfolio_img = "https://img.icons8.com/color/48/000000/domain--v1.png"
        linkedin_img = "https://img.icons8.com/color/48/000000/linkedin.png"
        github_img = "https://img.icons8.com/fluent/48/000000/github.png"
        youtube_img = "https://img.icons8.com/?size=50&id=19318&format=png"
        twitter_img = "https://img.icons8.com/color/48/000000/twitter.png"
        facebook_img = "https://img.icons8.com/color/48/000000/facebook-new.png"

        # Define destination URLs
        portfolio_url = "https://abu-usama.netlify.app/"
        linkedin_url = "https://www.linkedin.com/in/abu--usama"
        github_url = "https://github.com/UsamaMunawarr"
        youtube_url ="https://www.youtube.com/@CodeBaseStats"
        twitter_url = "https://twitter.com/Usama__Munawar?t=Wk-zJ88ybkEhYJpWMbMheg&s=09"
        facebook_url = "https://www.facebook.com/profile.php?id=100005320726463&mibextid=9R9pXO"

        # Interactive CSS for icons
        st.markdown("""
        <style>
        .social-icon {
            transition: all 0.3s ease;
            filter: drop-shadow(0 4px 6px rgba(0,0,0,0.2));
        }
        .social-icon:hover {
            transform: translateY(-8px) scale(1.1);
            filter: drop-shadow(0 8px 15px rgba(0,212,170,0.4));
        }
        </style>
        """, unsafe_allow_html=True)

        # Create clickable row of interactive icons
        st.markdown(
            f'<div style="display: flex; gap: 20px; margin-top: 15px;">'
            f'<a href="{portfolio_url}" target="_blank"><img class="social-icon" src="{portfolio_img}" width="55" height="55" title="My Portfolio"></a>'
            f'<a href="{github_url}" target="_blank"><img class="social-icon" src="{github_img}" width="55" height="55" title="GitHub"></a>'
            f'<a href="{linkedin_url}" target="_blank"><img class="social-icon" src="{linkedin_img}" width="55" height="55" title="LinkedIn"></a>'
            f'<a href="{youtube_url}" target="_blank"><img class="social-icon" src="{youtube_img}" width="55" height="55" title="YouTube"></a>'
            f'<a href="{twitter_url}" target="_blank"><img class="social-icon" src="{twitter_img}" width="55" height="55" title="Twitter"></a>'
            f'<a href="{facebook_url}" target="_blank"><img class="social-icon" src="{facebook_img}" width="55" height="55" title="Facebook"></a>'
            f'</div>',
            unsafe_allow_html=True
        )
