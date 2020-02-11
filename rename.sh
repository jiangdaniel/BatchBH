PLOTS=out

for SIZE in 10 100 1000
do
    for EXT in png pkl
    do
        cp $PLOTS/mean3-bbh/pi1_10_batch_${SIZE}_rdiff.$EXT $PLOTS/${EXT}s/rdiff$SIZE.$EXT
    done
done

for EXT in png pkl
do
    mkdir -p $PLOTS/${EXT}s
done


for TASK in mean3 mean0
do
    for TESTERS in bbh bsbh bh sbh
    do
        for EXT in png pkl
        do
            cp $PLOTS/$TASK-$TESTERS/pi1s.$EXT $PLOTS/${EXT}s/${TASK}_${TESTERS}_pi1s.$EXT
        done
    done
done


for TASK in mean3 mean0
do
    for TESTERS in bbh bsbh
    do
        for EXT in png pkl
        do
            cp $PLOTS/$TASK-$TESTERS/pi1_10.$EXT $PLOTS/${EXT}s/${TASK}_${TESTERS}_pi1_1.$EXT
            cp $PLOTS/$TASK-$TESTERS/pi1_50.$EXT $PLOTS/${EXT}s/${TASK}_${TESTERS}_pi1_5.$EXT
            cp $PLOTS/$TASK-$TESTERS/monotone.$EXT $PLOTS/${EXT}s/monotone_${TASK}_${TESTERS}.$EXT

            cp $PLOTS/$TASK-$TESTERS/pi1_10_rdiff.$EXT $PLOTS/${EXT}s/${TASK}_${TESTERS}_pi1_1_rdiff.$EXT
            cp $PLOTS/$TASK-$TESTERS/pi1_50_rdiff.$EXT $PLOTS/${EXT}s/${TASK}_${TESTERS}_pi1_5_rdiff.$EXT
        done
    done
done

