import plotly.graph_objects as go
import streamlit as st

from utility import handler

# Streamlit app
st.title("Sentence Embeddings Visualization with PCA")

# Input sentences
sentences = st.text_area(
    "Enter sentences (one per line):",
    value="""\
EGPU-3201_Datasheet: EGPU-3201 M.2 to dual USB 3.0 Module External power in *Needed when total power consumption exceed 2W 5V GND GND 5V 19pin header 20cm Power Cable Mechanical Drawing 25cm USB Cable Features −Alternative M.2 2260 or 2280 B-M key −Compliant with PCI Express Base Specification Revision 2.0 −Compliant with USB 3.0 Specification Revision 1.0, up to 5 Gbs −Compliant with xHCI 1.0 −Supports 2 USB 3.0 ports (Share PCIe Gen2 x1 bandwidth) −Supports each USB port output power up to 5V 900ma with external power in (200mA per port without external power) −Complies with EN61000-4-2 (ESD) Air-15kV, Contact-8kV −Optional Industrial Temperature (-40°C to +85°C) support USB 3.0 Pin Assignment Signal Pin Pin Signal Vbus IntA_P1_SSRX- IntA_P1_SSRX+ GND IntA_P1_SSTX- IntA_P1_SSTX+ GND IntA_P1_D- IntA_P1_D+ 1 2 3 4 5 6 7 8 9 10 19 18 17 16 15 14 13 12 11 Vbus IntA_P1_SSRX- IntA_P2_SSRX+ GND IntA_P2_SSTX- IntA_P2_SSTX+ GND IntA_P2_D- IntA_P2_D+ Headquarters (Taiwan) 5F., No.237, Sec. 1, Datong Rd., Xizhi Dist., New Taipei 
EGPU-3201_Datasheet: ports (Share PCIe Gen2 x1 bandwidth) −Supports each USB port output power up to 5V 900ma with external power in (200mA per port without external power) −Complies with EN61000-4-2 (ESD) Air-15kV, Contact-8kV −Optional Industrial Temperature (-40°C to +85°C) support USB 3.0 Pin Assignment Signal Pin Pin Signal Vbus IntA_P1_SSRX- IntA_P1_SSRX+ GND IntA_P1_SSTX- IntA_P1_SSTX+ GND IntA_P1_D- IntA_P1_D+ 1 2 3 4 5 6 7 8 9 10 19 18 17 16 15 14 13 12 11 Vbus IntA_P1_SSRX- IntA_P2_SSRX+ GND IntA_P2_SSTX- IntA_P2_SSTX+ GND IntA_P2_D- IntA_P2_D+ Headquarters (Taiwan) 5F., No.237, Sec. 1, Datong Rd., Xizhi Dist., New Taipei City 221, Taiwan Tel: +886-2-7703-3000 Email: sales@innodisk.com Branch Offices: USA usasales@innodisk.com +1-510-770-9421 Europe eusales@innodisk.com +31-40-3045-400 Japan jpsales@innodisk.com +81-3-6667-0161 China sales_cn@innodisk.com +86-755-2167-3689 www.innodisk.com © 2020 Innodisk Corporation. All right reserved. Specifications are subject to change without prior notice. November 24, 2022 Specifications Form-Factor M.2 2260/2280 B+M Key Input I/F PCI Express 2.0 Output I/F USB 3.0 Output Connector 19 Pin header Power Consumption MAX 1W (3.3V x 303mA) Dimension (WxLxH) 22.0 x 60.0 
EGPU-3201_Datasheet: City 221, Taiwan Tel: +886-2-7703-3000 Email: sales@innodisk.com Branch Offices: USA usasales@innodisk.com +1-510-770-9421 Europe eusales@innodisk.com +31-40-3045-400 Japan jpsales@innodisk.com +81-3-6667-0161 China sales_cn@innodisk.com +86-755-2167-3689 www.innodisk.com © 2020 Innodisk Corporation. All right reserved. Specifications are subject to change without prior notice. November 24, 2022 Specifications Form-Factor M.2 2260/2280 B+M Key Input I/F PCI Express 2.0 Output I/F USB 3.0 Output Connector 19 Pin header Power Consumption MAX 1W (3.3V x 303mA) Dimension (WxLxH) 22.0 x 60.0 x 8.4 mm | 22.0 x 80.0 x 8.4 mm Temperature Operation: STD: 0°C ~ +70°C. WT: -40°C ~ +85°C Storage: -55°C ~ +95°C Environment Vibration: 5G @7~2000Hz, Shock: 50G @ 0.5ms Notes Please download driver from MyInnodisk website. Windows: XP(32 bit), 7(32/64 bit), Vista Linux: Kernel v2.6.0, v2.6.8 *After Win8 and Linux Kernel v2.6.31 supports 
EGPU-3201_Datasheet: x 8.4 mm | 22.0 x 80.0 x 8.4 mm Temperature Operation: STD: 0°C ~ +70°C. WT: -40°C ~ +85°C Storage: -55°C ~ +95°C Environment Vibration: 5G @7~2000Hz, Shock: 50G @ 0.5ms Notes Please download driver from MyInnodisk website. Windows: XP(32 bit), 7(32/64 bit), Vista Linux: Kernel v2.6.0, v2.6.8 *After Win8 and Linux Kernel v2.6.31 supports built-in xHCI 1.0. Order Information EGPU-3201-C1 M.2 2260 to dual USB 3.0 Module EGPU-3201-C2 M.2 2280 to dual USB 3.0 Module EGPU-3201-W1 M.2 2260 to dual USB 3.0 Module W/T EGPU-3201-W2 M.2 2280 to dual USB 3.0 Module W/T www.innodisk.com
EGPS_3401_Datasheet: EGPS-3401 M.2 3042 to four SATA III Module J2 J3 SATAIII 7pin Male Features −PCI Express 2.0 to four SATA III ports. −Supports AHCI, Port-Multiplier. −Supports Native Command Queuing −Supports error reporting, recovery and correction. −30µ golden finger, 3 years warranty. −Industrial design, manufactured in innodisk Taiwan J1 J4 Mechanical Drawing Headquarters (Taiwan) 5F., No. 237, Sec. 1, Datong Rd., Xizhi Dist., New Taipei City 221, Taiwan Tel: +886-2-77033000 Email: sales@innodisk.com Branch Offices: USA usasales@innodisk.com +1-510-770-9421 Europe eusales@innodisk.com +31-40-3045-400 Japan jpsales@innodisk.com +81-3-6667-0161 China sales_cn@innodisk.com +86-755-21673689 www.innodisk.com © 2017 Innodisk Corporation. All right reserved. Specifications are subject to change without prior notice. June 28, 2023 Specifications Form-Factor M.2 3042-B-M Input I/F PCI Express 2.0 Output I/F SATA III Output Connector SATA 7pin x 4 Bridge Marvell 88SE9215 TDP 2.74W (3.3V x 830mA) Dimension (WxLxH) 30 x 42 x 13.8 
EGPS_3401_Datasheet: Taipei City 221, Taiwan Tel: +886-2-77033000 Email: sales@innodisk.com Branch Offices: USA usasales@innodisk.com +1-510-770-9421 Europe eusales@innodisk.com +31-40-3045-400 Japan jpsales@innodisk.com +81-3-6667-0161 China sales_cn@innodisk.com +86-755-21673689 www.innodisk.com © 2017 Innodisk Corporation. All right reserved. Specifications are subject to change without prior notice. June 28, 2023 Specifications Form-Factor M.2 3042-B-M Input I/F PCI Express 2.0 Output I/F SATA III Output Connector SATA 7pin x 4 Bridge Marvell 88SE9215 TDP 2.74W (3.3V x 830mA) Dimension (WxLxH) 30 x 42 x 13.8 mm, Weight: 7.5g Temperature Operation: STD: 0°C ~ +70°C Storage: -55°C ~ +95°C Environment Vibration: 5G @7~2000Hz, Shock: 50G @ 0.5ms Notes Order Information EGPS-3401-C1 M.2 to four SATA III Module www.innodisk.com\
""",
)
sentences = sentences.splitlines()

import pandas as pd

df = pd.DataFrame({"Sentence": sentences})
st.dataframe(df, use_container_width=True)
index_list = [f"{i}: {sentences[i][:20]}..." for i in range(len(sentences))]
# Input prompt
prompt = st.text_input("Enter a prompt:", value="What is EGPS-3401?")

# PCA
sentence_embeddings = handler.sentence_to_embedding(sentences=sentences + [prompt])
sentence_embeddings_2d = handler.sentence_embedding_to_2d(
    sentence_embeddings=sentence_embeddings
)
sentence_embeddings_2d, prompt_embedding_2d = (
    sentence_embeddings_2d[:-1],
    sentence_embeddings_2d[-1],
)

# Figure
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=sentence_embeddings_2d[:, 0],
        y=sentence_embeddings_2d[:, 1],
        mode="markers",
        text=index_list,
        name="Sentence",
    )
)
fig.add_trace(
    go.Scatter(
        x=[prompt_embedding_2d[0]],
        y=[prompt_embedding_2d[1]],
        mode="markers",
        text=prompt,
        name="Prompt",
    )
)
fig.update_traces(marker=dict(size=15))
fig.update_layout(
    title="2D PCA of Sentence Embeddings",
)

# Display the plot
st.plotly_chart(fig)
