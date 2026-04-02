set term pngcairo size 900,700 enhanced
set output "bayesian_regression.png"

set multiplot layout 2,2 title "Bayesian Linear Regression"

set xrange [-2:2]
set yrange [-15:25]
set grid

# style
set style line 1 lc rgb "black" lw 3   # mean thicker
set style line 2 lc rgb "red" lw 3     # variance thicker
set style line 3 lc rgb "blue" pt 7 ps 0.5   # smaller points


# 1. Ground truth
set title "Ground truth"
plot "ground_truth.dat" using 1:2 with lines ls 1 title "mean", \
     "ground_truth.dat" using 1:3 with lines ls 2 notitle, \
     "ground_truth.dat" using 1:4 with lines ls 2 notitle


# 2. Predict result (final)
set title "Predict result"
plot "points.dat" using 1:2 with points ls 3 title "data", \
     "predict_final.dat" using 1:2 with lines ls 1 title "mean", \
     "predict_final.dat" using 1:3 with lines ls 2 notitle, \
     "predict_final.dat" using 1:4 with lines ls 2 notitle


# 3. After 10 incomes
set title "After 10 incomes"
plot "predict_10.dat" using 1:2 with lines ls 1 title "mean", \
     "predict_10.dat" using 1:3 with lines ls 2 notitle, \
     "predict_10.dat" using 1:4 with lines ls 2 notitle, \
     "points.dat" every ::0::9 using 1:2 with points ls 3 title "data"


# 4. After 50 incomes
set title "After 50 incomes"
plot "predict_50.dat" using 1:2 with lines ls 1 title "mean", \
     "predict_50.dat" using 1:3 with lines ls 2 notitle, \
     "predict_50.dat" using 1:4 with lines ls 2 notitle, \
     "points.dat" every ::0::49 using 1:2 with points ls 3 title "data"

unset multiplot