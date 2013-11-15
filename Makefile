# simple makefile to simplify repetetive build env management tasks under posix
# inspired by https://github.com/scikit-learn/scikit-learn/blob/master/Makefile

PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests

all: clean inplace test

clean:
	$(PYTHON) setup.py clean
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext --inplace

#test-code: in
#	$(NOSETESTS) -s -v sklearn
#test-doc:
#	$(NOSETESTS) -s -v doc/ doc/modules/ doc/datasets/ \
#	doc/developers doc/tutorial/basic doc/tutorial/statistical_inference

#test-coverage:
#	rm -rf coverage .coverage
#	$(NOSETESTS) -s -v --with-coverage sklearn

#test: test-code test-doc

#trailing-spaces:
#	find sklearn -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

#cython:
#	find sklearn -name "*.pyx" | xargs $(CYTHON)


#doc: inplace
#	make -C doc html

#doc-noplot: inplace
#	make -C doc html-noplot

#code-analysis:
#	flake8 sklearn | grep -v __init__ | grep -v external
#	pylint -E -i y sklearn/ -d E1103,E0611,E1101

