echo "Setting up Enviroment variables."
export PYAUTO_PATH=/home/jammy/PycharmProjects/PyAuto/
export PYAUTOFIT_PATH=$PYAUTO_PATH/PyAutoFit
export WORKSPACE_PATH=$PYAUTO_PATH/autofit_workspace
export HOWTOFIT_PATH=$WORKSPACE_PATH/howtofit
export CHAPTER_PATH=$HOWTOFIT_PATH/chapter_1_introduction
export WORKSPACE=$WORKSPACE_PATH

RUN_SCRIPTS=TRUE

echo "Removing old notebooks."
rm -rf $CHAPTER_PATH/tutorial_*/.ipynb_checkpoints
rm $CHAPTER_PATH/tutorial_*/*.ipynb

if [ "$RUN_SCRIPTS" == "TRUE" ]; then

  echo "Running scripts."
  find $WORKSPACE_PATH/config -type f -exec sed -i 's/backend=default/backend=Agg/g' {} +

  python3 $CHAPTER_PATH/tutorial_1_model_mapping/tutorial_1_model_mapping.py
  python3 $CHAPTER_PATH/tutorial_2_model_fitting/tutorial_2_model_fitting.py
  python3 $CHAPTER_PATH/tutorial_3_non_linear_search/tutorial_3_non_linear_search.py
  python3 $CHAPTER_PATH/tutorial_4_source_code/tutorial_4_source_code.py
  python3 $CHAPTER_PATH/tutorial_5_visualization_masking/tutorial_5_visualization_masking.py
  python3 $CHAPTER_PATH/tutorial_6_complex_models/tutorial_6_complex_models.py
  python3 $CHAPTER_PATH/tutorial_7_phase_customization/tutorial_7_phase_customization.py
  python3 $CHAPTER_PATH/tutorial_8_aggregator/tutorial_8_aggregator_part_1.py
  python3 $CHAPTER_PATH/tutorial_8_aggregator/tutorial_8_aggregator_part_2.py

  find $WORKSPACE_PATH/config -type f -exec sed -i 's/backend=Agg/backend=default/g' {} +
fi

echo "Converting scripts to notebooks."
cd $CHAPTER_PATH/tutorial_1_model_mapping/
py_to_notebook $CHAPTER_PATH/tutorial_1_model_mapping/tutorial_1_model_mapping.py
cd $CHAPTER_PATH/tutorial_2_model_fitting
py_to_notebook $CHAPTER_PATH/tutorial_2_model_fitting/tutorial_2_model_fitting.py
cd $CHAPTER_PATH/tutorial_3_non_linear_search
py_to_notebook $CHAPTER_PATH/tutorial_3_non_linear_search/tutorial_3_non_linear_search.py
cd $CHAPTER_PATH/tutorial_4_source_code
py_to_notebook $CHAPTER_PATH/tutorial_4_source_code/tutorial_4_source_code.py
cd $CHAPTER_PATH/tutorial_5_visualization_masking
py_to_notebook $CHAPTER_PATH/tutorial_5_visualization_masking/tutorial_5_visualization_masking.py
cd $CHAPTER_PATH/tutorial_6_complex_models
py_to_notebook $CHAPTER_PATH/tutorial_6_complex_models/tutorial_6_complex_models.py
cd $CHAPTER_PATH/tutorial_7_phase_customization
py_to_notebook $CHAPTER_PATH/tutorial_7_phase_customization/tutorial_7_phase_customization.py
cd $CHAPTER_PATH/tutorial_8_aggregator
py_to_notebook $CHAPTER_PATH/tutorial_8_aggregator/tutorial_8_aggregator_part_1.py
cd $CHAPTER_PATH/tutorial_8_aggregator
py_to_notebook $CHAPTER_PATH/tutorial_8_aggregator/tutorial_8_aggregator_part_2.py

echo "Moving new notebooks to PyAutoFit/howtofens folder."
rm -rf $PYAUTOFIT_PATH/howtofit/*
cp -r $HOWTOFIT_PATH/config $PYAUTOFIT_PATH/howtofit
cp -r $HOWTOFIT_PATH/simulators $PYAUTOFIT_PATH/howtofit
cp -r $CHAPTER_PATH $PYAUTOFIT_PATH/howtofit/
cp $PYAUTOFIT_PATH/__init__.py $PYAUTOFIT_PATH/howtofit/
cp $PYAUTOFIT_PATH/__init__.py $PYAUTOFIT_PATH/howtofit/chapter_1_introduction

echo "Renaming import autofit_workspace to just howtofit for Sphinx build."
find $PYAUTOFIT_PATH/howtofit/ -type f -exec sed -i 's/from autofit_workspace./from /g' {} +

echo "Adding new notebooks to PyAutoFit GitHub repo."
cd $PYAUTOFIT_PATH
git add -f $PYAUTOFIT_PATH/howtofit/dataset/chapter_1
git add -f $PYAUTOFIT_PATH/howtofit/chapter_1_introduction

echo "returning to generate folder."
cd $HOWTOFIT_PATH/generate