#pragma once

#include <stdexcept>

using namespace std;


class InvalidOperationException : public logic_error
{
public:
	InvalidOperationException(void);
	InvalidOperationException(string errorMessage);
	virtual ~InvalidOperationException(void);
};