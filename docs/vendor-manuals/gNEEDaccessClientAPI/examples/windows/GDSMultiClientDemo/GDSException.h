#pragma once

#include <stdexcept>
#include <GDSClientAPI.h>

using namespace std;


class GDSException : public logic_error
{
public:
	GDSException(void);
	GDSException(GDS_RESULT result);
	GDSException(uint32_t errorCode, string errorMessage);
	virtual ~GDSException(void);

	uint32_t ErrorCode();
	string ErrorMessage();

private:
	uint32_t _errorCode;
};

