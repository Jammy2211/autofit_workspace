export WORKSPACE_PATH=/home/jammy/PycharmProjects/PyAuto/autofit_workspace
export EXAMPLE_PATH=$WORKSPACE_PATH/examples/

RUN_SCRIPTS=TRUE

find $WORKSPACE_PATH -type f -exec sed -i 's/backend = TKAgg/backend = TKAgg/g' {} +

if [ "$RUN_SCRIPTS" == "TRUE" ]; then

  python3 $EXAMPLE_PATH/simple/data.py
  python3 $EXAMPLE_PATH/simple/fit.py
  python3 $EXAMPLE_PATH/simple/result.py
  python3 $EXAMPLE_PATH/simple/aggregator.py
  python3 $EXAMPLE_PATH/complex/data.py
  python3 $EXAMPLE_PATH/complex/fit.py
  python3 $EXAMPLE_PATH/complex/result.py
  python3 $EXAMPLE_PATH/complex/aggregator.py

fi

cd $EXAMPLE_PATH/simple
py_to_notebook $EXAMPLE_PATH/simple/data.py
py_to_notebook $EXAMPLE_PATH/simple/fit.py
py_to_notebook $EXAMPLE_PATH/simple/result.py
py_to_notebook $EXAMPLE_PATH/simple/aggregator.py
cd $EXAMPLE_PATH/complex
py_to_notebook $EXAMPLE_PATH/complex/data.py
py_to_notebook $EXAMPLE_PATH/complex/fit.py
py_to_notebook $EXAMPLE_PATH/complex/result.py
py_to_notebook $EXAMPLE_PATH/complex/aggregator.py

find $WORKSPACE_PATH -type f -exec sed -i 's/backend = TKAgg/backend = TKAgg/g' {} +