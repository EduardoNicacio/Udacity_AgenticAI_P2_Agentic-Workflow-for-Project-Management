@echo off
rem -------------------------------------------------------------
rem  run_phase1_tests.bat
rem  Runs all Phase 1 agent tests with real API calls.
rem  Make sure you have:
rem    • Activated your virtual environment (e.g. call venv\Scripts\activate.bat)
rem    • Created a .env file within phase_1\ containing your OPENAI_API_KEY
rem -------------------------------------------------------------

echo ======================================================
echo Running All Phase 1 Agent Tests (with actual API calls)
echo ======================================================
echo.
echo Make sure you have:
echo   1. Activated your virtual environment
echo   2. Created a .env file in phase_1\ with your OPENAI_API_KEY
echo.

rem Wait for the user to press Enter (or Ctrl+C to abort)
set /p dummy=Press Enter to continue or Ctrl+C to cancel...

cls

rem Change into the phase_1 directory
cd /d phase_1

rem -------------------------------------------------------------
rem  Test 1: DirectPromptAgent
rem -------------------------------------------------------------
echo.
echo ======================================================
echo Test 1: DirectPromptAgent
echo ======================================================
python direct_prompt_agent.py
echo.
set /p dummy=Press Enter to continue to next test...

rem -------------------------------------------------------------
rem  Test 2: AugmentedPromptAgent
rem -------------------------------------------------------------
echo.
echo ======================================================
echo Test 2: AugmentedPromptAgent
echo ======================================================
python augmented_prompt_agent.py
echo.
set /p dummy=Press Enter to continue to next test...

rem -------------------------------------------------------------
rem  Test 3: KnowledgeAugmentedPromptAgent
rem -------------------------------------------------------------
echo.
echo ======================================================
echo Test 3: KnowledgeAugmentedPromptAgent
echo ======================================================
python knowledge_augmented_prompt_agent.py
echo.
set /p dummy=Press Enter to continue to next test...

rem -------------------------------------------------------------
rem  Test 4: RAGKnowledgePromptAgent
rem -------------------------------------------------------------
echo.
echo ======================================================
echo Test 4: RAGKnowledgePromptAgent
echo ======================================================
python rag_knowledge_prompt_agent.py
echo.
set /p dummy=Press Enter to continue to next test...

rem -------------------------------------------------------------
rem  Test 5: EvaluationAgent
rem -------------------------------------------------------------
echo.
echo ======================================================
echo Test 5: EvaluationAgent
echo ======================================================
python evaluation_agent.py
echo.
set /p dummy=Press Enter to continue to next test...

rem -------------------------------------------------------------
rem  Test 6: RoutingAgent
rem -------------------------------------------------------------
echo.
echo ======================================================
echo Test 6: RoutingAgent
echo ======================================================
python routing_agent.py
echo.
set /p dummy=Press Enter to continue to next test...

rem -------------------------------------------------------------
rem  Test 7: ActionPlanningAgent
rem -------------------------------------------------------------
echo.
echo ======================================================
echo Test 7: ActionPlanningAgent
echo ======================================================
python action_planning_agent.py
echo.

rem -------------------------------------------------------------
rem  All tests finished
rem -------------------------------------------------------------
echo.
echo ======================================================
echo All Phase 1 Tests Completed!
echo ======================================================
echo.
echo Please review the outputs and capture screenshots or save to text files
echo for your project submission.
echo.
pause >nul   rem Optional: keep window open until a key is pressed
