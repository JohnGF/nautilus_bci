#include "stdafx.h"
#include "InvalidOperationException.h"


InvalidOperationException::InvalidOperationException(void)
	: logic_error("The operation is invalid for the object's current state.")
{
}

InvalidOperationException::InvalidOperationException(string errorMessage)
	: logic_error(errorMessage)
{
}

InvalidOperationException::~InvalidOperationException(void)
{
}