#include "StdAfx.h"
#include "GDSException.h"


GDSException::GDSException(void)
	: logic_error("An error occurred while communicating with the GDS."),
	_errorCode(GDS_ERROR_UNKNOWN)
{
}

GDSException::GDSException(GDS_RESULT result)
	: logic_error(result.ErrorMessage),
	_errorCode(result.ErrorCode)
{
}

GDSException::GDSException(uint32_t errorCode, string errorMessage)
	: logic_error(errorMessage),
	_errorCode(errorCode)
{
}

GDSException::~GDSException(void)
{
}

uint32_t GDSException::ErrorCode()
{
	return _errorCode;
}

string GDSException::ErrorMessage()
{
	return string(what());
}