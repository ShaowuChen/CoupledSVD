#!/bin/bash
for i in $(seq 0 7);do
    chmod +x run$i.sh&
    sh run$i.sh&
done