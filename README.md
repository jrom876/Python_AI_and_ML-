# Python_AI_and_ML-
The co2_data directory contains Python AI/ML/NN code that extracts and processes raw CO2 data from the serial port of an Arduino K30 CO2 sensor node.

The main CO2 file -- co2_data/read_serial.py -- contains code to read CO2 sensor data, make predictions, and display data plots using various AI/ML/NN algorithms and Python data science libraries.

TODO:  Finish the nRF24L01 handler for an RPi 5 B+ platform; I am using the Arduino USB serial port data for now, but I would rather get sensor data wirelessly using the nRF24L01 in wifi mode using the MQTT protocol.

TODO:  Include the Arduino K30 code and hardware requirements.

Words of Wisdom: 

"I've seen a very typical pattern of errors among interns and new grads:

    Their changes are much too large - hundreds of lines all sent for review at once, 
        rather than splitting up the changes into separable units when possible.
    They leave in commented-out code and debug statements.
    They don't use functions to organize their code logically.
    They use magic numbers without comments.
    Their function and variable names are terrible (“whyIsThisNameSoLongAndItJustKeepsGoing”, “tempVar1”, etc.).
    They don't use classes and interfaces correctly.
    They don't write unit tests at all.
    Their changelist comments are unhelpful (“Fix bugs”, “First check in”, etc).

They usually learn fast, though!"
