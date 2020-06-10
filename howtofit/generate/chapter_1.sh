export PYAUTO_PATH=/home/jammy/PycharmProjects/PyAuto/
export WORKSPACE_PATH=$PYAUTO_PATH/autofit_workspace
export CHAPTER1_PATH=$WORKSPACE_PATH/howtofit/chapter_1_introduction
export WORKSPACE=$WORKSPACE_PATH

RUN_SCRIPTS=FALSE

find $WORKSPACE_PATH -type f -exec sed -i 's/backend = TKAgg/backend = TKAgg/g' {} +

rm -rf $CHAPTER1_PATH/tutorial_*/.ipynb_checkpoints
rm $CHAPTER1_PATH/tutorial_*/.ipynb

if [ "$RUN_SCRIPTS" == "TRUE" ]; then

  python3 $CHAPTER1_PATH/tutorial_1_model_mapping/tutorial_1_model_mapping.py
  python3 $CHAPTER1_PATH/tutorial_2_model_fitting/tutorial_2_model_fitting.py
  python3 $CHAPTER1_PATH/tutorial_3_non_linear_search/tutorial_3_non_linear_search.py
  python3 $CHAPTER1_PATH/tutorial_4_source_code/tutorial_4_source_code.py
  python3 $CHAPTER1_PATH/tutorial_5_visualization_masking/tutorial_5_visualization_masking.py
  python3 $CHAPTER1_PATH/tutorial_6_complex_models/tutorial_6_complex_models.py
  python3 $CHAPTER1_PATH/tutorial_7_phase_customization/tutorial_7_phase_customization.py
  python3 $CHAPTER1_PATH/tutorial_8_aggregator/tutorial_8_aggregator_part_1.py
  python3 $CHAPTER1_PATH/tutorial_8_aggregator/tutorial_8_aggregator_part_2.py

fi

cd $CHAPTER1_PATH/tutorial_1_model_mapping/
py_to_notebook $CHAPTER1_PATH/tutorial_1_model_mapping/tutorial_1_model_mapping.py
cd $CHAPTER1_PATH/tutorial_2_model_fitting
py_to_notebook $CHAPTER1_PATH/tutorial_2_model_fitting/tutorial_2_model_fitting.py
cd $CHAPTER1_PATH/tutorial_3_non_linear_search
py_to_notebook $CHAPTER1_PATH/tutorial_3_non_linear_search/tutorial_3_non_linear_search.py
cd $CHAPTER1_PATH/tutorial_4_source_code
py_to_notebook $CHAPTER1_PATH/tutorial_4_source_code/tutorial_4_source_code.py
cd $CHAPTER1_PATH/tutorial_5_visualization_masking
py_to_notebook $CHAPTER1_PATH/tutorial_5_visualization_masking/tutorial_5_visualization_masking.py
cd $CHAPTER1_PATH/tutorial_6_complex_models
py_to_notebook $CHAPTER1_PATH/tutorial_6_complex_models/tutorial_6_complex_models.py
cd $CHAPTER1_PATH/tutorial_7_phase_customization
py_to_notebook $CHAPTER1_PATH/tutorial_7_phase_customization/tutorial_7_phase_customization.py
cd $CHAPTER1_PATH/tutorial_8_aggregator
py_to_notebook $CHAPTER1_PATH/tutorial_8_aggregator/tutorial_8_aggregator_part_1.py
cd $CHAPTER1_PATH/tutorial_8_aggregator
py_to_notebook $CHAPTER1_PATH/tutorial_8_aggregator/tutorial_8_aggregator_part_2.py

find $WORKSPACE_PATH -type f -exec sed -i 's/backend = TKAgg/backend = TKAgg/g' {} +