# Nautilus BCI

Welcome to the Nautilus BCI project. This suite serves as a multimodal hub for Brain-Computer Interface research. Device repositories and bridges (like g.Nautilus, Galaxy Watch, Artinis fNIRS) publish Lab Streaming Layer (LSL) streams, and Nautilus strictly acts as a consumer of these streams, integrating data seamlessly without needing to import device-specific code. LSL provides automatic discovery and a shared time base, ensuring hardware streams, task markers, and experimental environments operate under a unified synchronization protocol before being stored cleanly into Multimodal BIDS format.

**Important First Step**: Please ensure you configure your system properly by following the [LSL Network Setup](guides/lsl_network_setup.md) guide.

## System Architecture Data Flow

```mermaid
flowchart TD
    classDef device fill:#f9d0c4,stroke:#e88365,stroke-width:2px,color:black;
    classDef bridge fill:#c4e1f9,stroke:#5c9ad6,stroke-width:2px,color:black;
    classDef lsl fill:#f9f5c4,stroke:#d6c45c,stroke-width:2px,color:black;
    classDef processing fill:#d6f9c4,stroke:#7cd65c,stroke-width:2px,color:black;
    classDef output fill:#e2c4f9,stroke:#b15cd6,stroke-width:2px,color:black;

    Device1["EEG (e.g. g.Nautilus)"]:::device
    Device2["Watch (IMU/PPG)"]:::device
    Device3["fNIRS (Artinis)"]:::device
    TaskMarkers["Task Markers (base_task.py)"]:::device

    Bridge1["EEG Bridge"]:::bridge
    Bridge2["Watch Bridge"]:::bridge
    Bridge3["fNIRS Bridge"]:::bridge

    LSL(("Lab Streaming Layer (LSL)\nShared Clock & Discovery")):::lsl

    Recorder["BIDS Recorders\n(e.g. multimodal_bids_recorder.py)"]:::processing
    Dashboard["Real-Time Dashboards"]:::processing
    Decoder["Real-Time Decoders"]:::processing

    Games["Games & Assistive Output\n(via UDP -> LSL Bridge)"]:::output

    Device1 --> Bridge1
    Device2 --> Bridge2
    Device3 --> Bridge3

    Bridge1 --> LSL
    Bridge2 --> LSL
    Bridge3 --> LSL
    TaskMarkers --> LSL

    LSL --> Recorder
    LSL --> Dashboard
    LSL --> Decoder

    Decoder --> Games
    Games -->|markers via UDP->LSL bridge| LSL
```
